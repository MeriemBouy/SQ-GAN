import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.losses.lpips import LPIPS
from modules.discriminator.model import NLayerDiscriminator, weights_init
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


def adopt_weight(weight, global_step, threshold=0, value=0.):
    """Zero-out `weight` if `global_step < threshold`."""
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake))
    )
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    """
    A single class that can handle:
     - mode="ssm":   uses cross-entropy on semantic reconstruction, D sees (semantic vs. predicted_sem).
     - mode="img":   uses L1+LPIPS on image, D sees (real_img vs. pred_img).
     - mode="all":   same as "img" path; only the image pipeline is updated.
    """
    def __init__(
        self,
        mode: str,                    # "ssm", "img", or "all"
        disc_start: int,
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_lpips_weight: float = 1.0,
        label_weight: float = None,
        label_weights: list = None,
        use_actnorm: bool = False,
        disc_conditional: bool = False,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        disc_weight_max: float = None,
        class_subdivisions=None,
        # For SSM mode only:
        semantic_weights=None,
    ):
        super().__init__()
        self.mode = mode.lower().strip()  # "ssm", "img", or "all"
        assert self.mode in ["ssm", "img", "all"], \
            f"mode must be one of ['ssm','img','all'], got '{self.mode}'."

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_lpips_weight = perceptual_lpips_weight
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_weight_max = disc_weight_max

        # If "img" or "all"
        if self.mode in ["img", "all"]:
            # Check if label_weight, label_weights are provided and create the weighting for the L1 portion.
            if label_weight is not None and label_weights is not None:
                self.label_weight_tensor = (
                    torch.tensor(label_weights) * label_weight
                )
            else:
                self.label_weight_tensor = None

            # For LPIPS
            if perceptual_lpips_weight > 0:
                self.perceptual_loss = LPIPS().eval()
            else:
                self.perceptual_loss = None

        else:
            self.label_weight_tensor = None
            self.perceptual_loss = None

        # If "ssm" => initialize the stuff 
        if self.mode == "ssm":
            # cross-entropy on the semantic output
            if semantic_weights is not None:
                self.semantic_loss = nn.CrossEntropyLoss(
                    weight=torch.tensor(semantic_weights, dtype=torch.float32)
                )
            else:
                self.semantic_loss = nn.CrossEntropyLoss()
        else:
            self.semantic_loss = None

        # ---------------------------------------------------------------------
        # Discriminator setup
        # ---------------------------------------------------------------------
        assert disc_loss in ["hinge", "vanilla"], f"Unknown disc_loss={disc_loss}"
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        else:
            self.disc_loss = vanilla_d_loss

        print(f"[VQLPIPSWithDiscriminatorUnified] mode={self.mode}, disc_loss={disc_loss}")

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)

        self.class_subdivisions = class_subdivisions  # if you need them

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Implements the adaptive weight for the generator loss:
        A = ||grad_{z}(nll)|| / (||grad_{z}(g)|| + eps).
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # fallback - not recommended
            raise ValueError("Please pass the 'last_layer' param for adaptive weight calculation.")

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        # optional: limit maximum
        if self.disc_weight_max is not None:
            d_weight = torch.clamp(d_weight, max=self.disc_weight_max)

        return d_weight

    def forward(
        self,
        codebook_loss,
        semantic,         # [B, C, H, W], or the ground truth mask
        inputs,           # real image if mode in ["img","all"]
        reconstructions,  # predicted image or predicted semantic
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train"
    ):
        """
        Unified forward. Behavior depends on `self.mode`.
        """
        # ---------------------------------------------------------------------
        # 1) Reconstruction Loss
        # ---------------------------------------------------------------------
        if self.mode == "ssm":
            # CrossEntropyLoss between reconstructions (logits for K classes)
            # and argmax(semantic) as target
            #   reconstructions: [B, classes, H, W]
            #   semantic:        [B, classes, H, W]
            # We interpret the GT as the argmax across channels
            ce_target = torch.argmax(semantic, dim=1)  # [B, H, W]
            rec_loss = self.semantic_loss(reconstructions, ce_target)
            nll_loss = torch.mean(rec_loss)

            # For the discriminator, real => semantic, fake => reconstructions
            real_data = semantic
            fake_data = reconstructions

        else:
            # "img" or "all" => use L1 + optional LPIPS
            #  - Weighted by label_weight_tensor if provided
            #  - Additional partial “semantic-based mask” if used in your original code

            # Basic L1
            l1 = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.label_weight_tensor is not None:
                # Rescaling the l1 by the class given by the SSM
                sem_scaling = semantic[:, :-1, :, :] * self.label_weight_tensor.view(1, -1, 1, 1).to(semantic.device)
                sem_scaling = sem_scaling.sum(dim=1, keepdim=True)
                l1 = sem_scaling * l1

            # optional LPIPS
            if self.perceptual_lpips_weight > 0 and self.perceptual_loss is not None:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = l1 + self.perceptual_lpips_weight * p_loss
            else:
                p_loss = torch.tensor(0.0, device=inputs.device)
                rec_loss = l1

            nll_loss = torch.mean(rec_loss)

            # For the semantic-aware discriminator rescale the classes and compare, real => real image, fake => reconstructed
            mask_discriminator = 0.50 * semantic[:, 8, :, :] \
                                 + semantic[:, 8, :, :] \
                                 + semantic[:, 10, :, :]
            # shape: [B, H, W]
            reconstructions_scaled = reconstructions + 0.85 * mask_discriminator.unsqueeze(1) * (inputs - reconstructions)

            # real vs fake
            real_data = inputs
            fake_data = reconstructions_scaled

        # ---------------------------------------------------------------------
        # 2) Generator or Discriminator branch
        # ---------------------------------------------------------------------
        if optimizer_idx == 0:
            # ----------------------
            # GENERATOR UPDATE
            # ----------------------
            # forward pass for the generator
            if cond is None:
                assert not self.disc_conditional, \
                    "disc_conditional=True but cond=None? Check your code or pass cond."
                logits_fake = self.discriminator(fake_data.contiguous())
            else:
                # conditional
                #   ssm => cat( [fake_sem, cond], dim=1 )
                #   img => cat( [fake_img, cond], dim=1 )
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((fake_data.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            # adaptive weight
            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)

            # total generator loss
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {
                f"{split}_total_loss": loss.detach().mean(),
                f"{split}_quant_loss": codebook_loss.detach().mean(),
                f"{split}_nll_loss":   nll_loss.detach().mean(),
                f"{split}_g_loss":     g_loss.detach().mean(),
                f"{split}_d_weight":   d_weight.detach(),
                f"{split}_disc_factor": torch.tensor(disc_factor, device=inputs.device)
            }

            # If "img" or "all", we might log rec_loss or p_loss explicitly:
            if self.mode in ["img", "all"]:
                log[f"{split}_rec_loss"] = rec_loss.detach().mean()
                if self.perceptual_lpips_weight > 0:
                    log[f"{split}_p_loss"] = p_loss.detach().mean()
            else:
                # "ssm"
                log[f"{split}_rec_loss"] = rec_loss.detach().mean()

            return loss, log

        elif optimizer_idx == 1:
            # ----------------------
            # DISCRIMINATOR UPDATE
            # ----------------------
            if cond is None:
                logits_real = self.discriminator(real_data.contiguous().detach())
                logits_fake = self.discriminator(fake_data.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((real_data.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((fake_data.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}_disc_loss": d_loss.detach().mean(),
                f"{split}_logits_real": logits_real.detach().mean(),
                f"{split}_logits_fake": logits_fake.detach().mean(),
            }

            return d_loss, log
