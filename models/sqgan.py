import torch
import os
import pytorch_lightning as pl
from utils.utils import instantiate_from_config
from modules.masked_quantization.tools import build_score_image
from modules.training_label_enhancer.label_enhancer import LabelEnhancer
from models.utils import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
import copy


class SQGAN(pl.LightningModule):
    def __init__(self,
                 mode: str,  # "ssm", "img", or "all"
                 # --- common to all modes ---
                 ckpt_path=None,
                 ckpt_path_parts=None,
                 ignore_keys=[],
                 monitor=None,
                 warmup_epochs=0,
                 scheduler_type="linear-warmup_cosine-decay",
                 # --- semantic configuration ---
                 encoder_semantic_config=None,
                 decoder_semantic_config=None,
                 masker_semantic_config=None,
                 demasker_semantic_config=None,
                 lossconfig_semantic=None,
                 vqconfig_semantic=None,
                 
                 # --- image configuration ---
                 encoder_config=None,
                 decoder_config=None,
                 masker_config=None,
                 demasker_config=None,
                 lossconfig=None,
                 vqconfig=None
                 ):
        """
        A unified model that can handle:
           mode="ssm" : Train semantic path only.
           mode="img" : Train image path only.
           mode="all" : Finetune image path conditioned on semantic path.
        """
        super().__init__()
        self.mode = mode.lower().strip()  # "ssm", "img", or "all"

        self.monitor = monitor
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type

        # ---------------------------------------------------------------------
        # Set up the semantic path components (used if mode in ["ssm","all"])
        # ---------------------------------------------------------------------
        if self.mode in ["ssm", "all"]:
            self.encoder_semantic = instantiate_from_config(encoder_semantic_config)
            self.decoder_semantic = instantiate_from_config(decoder_semantic_config)
            self.masker_semantic  = instantiate_from_config(masker_semantic_config)
            self.demasker_semantic = instantiate_from_config(demasker_semantic_config)
            self.quantize_semantic = instantiate_from_config(vqconfig_semantic)
        else:
            self.encoder_semantic = None
            self.decoder_semantic = None
            self.masker_semantic  = None
            self.demasker_semantic = None
            self.quantize_semantic = None

        # ---------------------------------------------------------------------
        # Set up the image path components (used if mode in ["img","all"])
        # ---------------------------------------------------------------------
        if self.mode in ["img", "all"]:
            self.encoder  = instantiate_from_config(encoder_config)
            self.decoder  = instantiate_from_config(decoder_config)
            self.masker   = instantiate_from_config(masker_config)
            self.demasker = instantiate_from_config(demasker_config)
            self.quantize = instantiate_from_config(vqconfig)
        else:
            self.encoder  = None
            self.decoder  = None
            self.masker   = None
            self.demasker = None
            self.quantize = None

        # ---------------------------------------------------------------------
        # Set up the rest
        # ---------------------------------------------------------------------
        if self.mode == "ssm":
            # Deep copy to avoid modifying the original config
            losscfg_semantic = copy.deepcopy(lossconfig_semantic)
            # Inject the mode parameter
            losscfg_semantic['params']['mode'] = self.mode
            self.loss = instantiate_from_config(losscfg_semantic)
        else:
            losscfg = copy.deepcopy(lossconfig)
            # Inject the mode parameter
            losscfg['params']['mode'] = self.mode
            self.loss = instantiate_from_config(losscfg)
        self.labelenhancer = LabelEnhancer()

        # ---------------------------------------------------------------------
        # Load checkpoint if provided
        # ---------------------------------------------------------------------
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if ckpt_path_parts is not None:
            # If your partial ckpts follow a naming pattern, 
            # you can load them with something like: 
            
            for ckpt_file in os.listdir(ckpt_path_parts):
                if not ckpt_file.endswith(".ckpt"):
                    continue
                attr_name = ckpt_file.split(".")[0]  # e.g. "encoder_semantic"
                full_path = os.path.join(ckpt_path_parts, ckpt_file)
                if getattr(self, attr_name, None) is not None:
                    sd = torch.load(full_path, map_location="cpu")
                    getattr(self, attr_name).load_state_dict(sd, strict=False)
                    print(f"Loaded partial checkpoint for {attr_name} from {ckpt_file}")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # -------------------------------------------------------------------------
    # Configuration of optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate

        # We always optimize the components that exist, depending on mode
        parameters_to_optimize = []
        if self.mode in ["ssm", "all"]:
            parameters_to_optimize += list(self.encoder_semantic.parameters())
            parameters_to_optimize += list(self.decoder_semantic.parameters())
            parameters_to_optimize += list(self.masker_semantic.parameters())
            parameters_to_optimize += list(self.demasker_semantic.parameters())
            parameters_to_optimize += list(self.quantize_semantic.parameters())

        if self.mode in ["img", "all"]:
            parameters_to_optimize += list(self.encoder.parameters())
            parameters_to_optimize += list(self.decoder.parameters())
            parameters_to_optimize += list(self.masker.parameters())
            parameters_to_optimize += list(self.demasker.parameters())
            parameters_to_optimize += list(self.quantize.parameters())

        opt_ae = torch.optim.Adam(parameters_to_optimize, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        # Warmup steps for scheduling
        warmup_steps = self.steps_per_epoch * self.warmup_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt_ae, Scheduler_LinearWarmup(warmup_steps)
                ),
                "interval": "step",
                "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt_disc, Scheduler_LinearWarmup(warmup_steps)
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt_ae,
                    Scheduler_LinearWarmup_CosineDecay(
                        warmup_steps=warmup_steps,
                        max_steps=self.training_steps,
                        multipler_min=multipler_min
                    )
                ),
                "interval": "step",
                "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt_disc,
                    Scheduler_LinearWarmup_CosineDecay(
                        warmup_steps=warmup_steps,
                        max_steps=self.training_steps,
                        multipler_min=multipler_min
                    )
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError(f"Unknown scheduler_type: {self.scheduler_type}")

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # Adapt channel if needed
        return x

    def get_last_layer(self):
        """
        By default, used by the patch discriminator; 
        might differ if you want to pick a different 'last layer' 
        depending on mode. 
        Here, we will always return the image decoder's last layer if it exists,
        else the semantic decoder's last layer.
        """
        if self.mode in ["img", "all"] and self.decoder is not None:
            return self.decoder.conv_out.weight
        elif self.mode in ["ssm", "all"] and self.decoder_semantic is not None:
            return self.decoder_semantic.conv_out.weight
        else:
            # fallback if something changes
            raise ValueError("No valid decoder found for get_last_layer().")

    # -------------------------------------------------------------------------
    # Forward paths depending on mode
    # -------------------------------------------------------------------------
    def forward_ssm(self, semantic):
        """
        Forward pass for semantic-only path (like SQGAN_ssm).
        Returns: xrec_sem, codebook_loss, ...
        """
        # 1) Encode
        h_sem = self.encoder_semantic(semantic)
        mask_out_sem = self.masker_semantic(h_sem, semantic[:, :-1, :, :])
        quant_sem = mask_out_sem["sample_features"]
        quant_sem, emb_loss_sem, info_sem = self.quantize_semantic(quant_sem)

        # 2) Importance based selection
        s_len = mask_out_sem["sampled_length"]
        sampled_sem = quant_sem[:, :, :s_len]
        remain_sem  = quant_sem[:, :, s_len:]
        s_idx  = mask_out_sem["sample_index"]
        r_idx  = mask_out_sem["remain_index"]
        mask_s = mask_out_sem["squeezed_mask"]

        # 3) Decode
        h_sem_dec = self.demasker_semantic(sampled_sem, remain_sem, s_idx, r_idx, mask_s)
        xrec_sem  = self.decoder_semantic(h_sem_dec)

        return xrec_sem, emb_loss_sem, info_sem, mask_out_sem

    def forward_img(self, x, semantic):
        """
        Forward pass for image-only path (like SQGAN_img).
        Returns: xrec_img, codebook_loss, ...
        """
        # 1) Encode
        h = self.encoder(x, semantic[:, :-1, :, :])
        mask_out = self.masker(h, semantic[:, :-1, :, :])
        quant = mask_out["sample_features"]
        quant, emb_loss, info = self.quantize(quant)

        # 2) Importance based selection
        s_len = mask_out["sampled_length"]
        sampled = quant[:, :, :s_len]
        remain  = quant[:, :, s_len:]
        s_idx  = mask_out["sample_index"]
        r_idx  = mask_out["remain_index"]
        mask_t = mask_out["squeezed_mask"]
        
        # 3) Decode
        h_dec = self.demasker(sampled, remain, s_idx, r_idx, mask_t)
        xrec_img = self.decoder(h_dec, [semantic[:, :-1, :, :] for _ in range(5)])

        return xrec_img, emb_loss, info, mask_out

    def forward_all(self, x, semantic):
        """
        Forward pass for mode="all" (like SQGAN_all).
        Returns: xrec_img, codebook_loss (image), info_img, xrec_sem, ...
        """
        # 1) Image path
        h_img = self.encoder(x, semantic[:, :-1, :, :])
        mask_out_img = self.masker(h_img, semantic[:, :-1, :, :])
        quant_img = mask_out_img["sample_features"]
        quant_img, emb_loss_img, info_img = self.quantize(quant_img)

        # 2) Semantic path
        h_sem = self.encoder_semantic(semantic)
        mask_out_sem = self.masker_semantic(h_sem, semantic[:, :-1, :, :])
        quant_sem = mask_out_sem["sample_features"]
        quant_sem, emb_loss_sem, info_sem = self.quantize_semantic(quant_sem)

        # 3) Importance based selection
        s_len_sem = mask_out_sem["sampled_length"]
        sampled_sem = quant_sem[:, :, :s_len_sem]
        remain_sem  = quant_sem[:, :, s_len_sem:]
        s_idx_sem  = mask_out_sem["sample_index"]
        r_idx_sem  = mask_out_sem["remain_index"]
        mask_sem   = mask_out_sem["squeezed_mask"]

        # 4) Decode semantic
        h_dec_sem = self.demasker_semantic(sampled_sem, remain_sem, s_idx_sem, r_idx_sem, mask_sem)
        xrec_sem  = self.decoder_semantic(h_dec_sem)

        # Convert that semantic reconstruction to one-hot for conditioning the image
        semantic_pred = torch.argmax(xrec_sem, dim=1)  # B,H,W
        one_hot_sem = torch.zeros_like(xrec_sem, dtype=torch.float32)  # B, nc, H, W
        one_hot_sem.scatter_(1, semantic_pred.unsqueeze(1), 1.0)

        # 5) Decode image (condition on predicted semantic)
        s_len_img = mask_out_img["sampled_length"]
        sampled_img = quant_img[:, :, :s_len_img]
        remain_img  = quant_img[:, :, s_len_img:]
        s_idx_img  = mask_out_img["sample_index"]
        r_idx_img  = mask_out_img["remain_index"]
        mask_img   = mask_out_img["squeezed_mask"]

        h_dec_img = self.demasker(sampled_img, remain_img, s_idx_img, r_idx_img, mask_img)
        # Condition the decoder on the predicted semantic (except the last channel)
        xrec_img  = self.decoder(h_dec_img, [one_hot_sem[:, :-1, :, :] for _ in range(5)])

        # We'll return only the image path codebook loss if you follow the 
        # same structure as SQGAN_all (which used "qloss" from the image side).
        return xrec_img, emb_loss_img, info_img, xrec_sem, emb_loss_sem, info_sem, mask_out_img, mask_out_sem

    # -------------------------------------------------------------------------
    # The main forward() calls the correct function
    # -------------------------------------------------------------------------
    def forward(self, x, label, semantic):
        if self.mode == "ssm":
            xrec_sem, emb_loss_sem, info_sem, _ = self.forward_ssm(semantic)
            return xrec_sem, emb_loss_sem, info_sem, None  # no image rec
        elif self.mode == "img":
            xrec_img, emb_loss_img, info_img, _ = self.forward_img(x, semantic)
            return xrec_img, emb_loss_img, info_img, None  # no semantic rec
        elif self.mode == "all":
            xrec_img, emb_loss_img, info_img, xrec_sem, emb_loss_sem, info_sem, _, _ = self.forward_all(x, semantic)
            # In the original SQGAN_all, only the image codebook loss is used for the training step
            return xrec_img, emb_loss_img, info_img, xrec_sem
        else:
            raise ValueError(f"Unknown mode {self.mode}, must be 'ssm', 'img', or 'all'.")

    # -------------------------------------------------------------------------
    # Training / validation steps
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, "image")
        label = self.get_input(batch, "label")
        semantic = self.get_input(batch, "semantic")

        # Data Augmentation that increase the presence of labeled elements (traffic signs and traffic lights) 
        label, x, semantic = self.labelenhancer.process_masks(label, x, semantic)

        # Forward step
        outputs = self(x, label, semantic)
        xrec, qloss = outputs[0], outputs[1]
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(
                codebook_loss=qloss,
                semantic=semantic,
                inputs=x,
                reconstructions=xrec,
                optimizer_idx=optimizer_idx,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train"
            )

            self.log("train_aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
            rec_loss = log_dict_ae.get("train_rec_loss", None)
            if rec_loss is not None:
                self.log("train_rec_loss", rec_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                del log_dict_ae["train_rec_loss"]
            self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=True)
            return aeloss

        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                codebook_loss=qloss,
                semantic=semantic,
                inputs=x,
                reconstructions=xrec,
                optimizer_idx=optimizer_idx,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train"
            )
            self.log("train_discloss", discloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        label = self.get_input(batch, "label")
        semantic = self.get_input(batch, "semantic")

        # Data Augmentation that increase the presence of labeled elements (traffic signs and traffic lights) 
        label, x, semantic = self.labelenhancer.process_masks(label, x, semantic)
        
        # Forward step
        outputs = self(x, label, semantic)
        xrec, qloss = outputs[0], outputs[1]

        aeloss, log_dict_ae = self.loss(
            codebook_loss=qloss,
            semantic=semantic,
            inputs=x,
            reconstructions=xrec,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val"
        )
        discloss, log_dict_disc = self.loss(
            codebook_loss=qloss,
            semantic=semantic,
            inputs=x,
            reconstructions=xrec,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val"
        )
        rec_loss = log_dict_ae.get("val_rec_loss", None)
        if rec_loss is not None:
            self.log("val_rec_loss", rec_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["val_rec_loss"]
        self.log("val_aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    # -------------------------------------------------------------------------
    # log_images
    # -------------------------------------------------------------------------
    def log_images(self, batch, **kwargs):
        split = kwargs.get('split', 'train')
        x = self.get_input(batch, "image").to(self.device)
        label = self.get_input(batch, "label").to(self.device)
        semantic = self.get_input(batch, "semantic").to(self.device)

        if split == "train":
            label, x, semantic = self.labelenhancer.process_masks(label, x, semantic)

        log = dict()

        # Depending on mode, we run the forward pass that gets the internal states
        # Then we create images for logging.
        if self.mode == "ssm":
            # forward_ssm
            xrec_sem, _, info_sem, mask_out_sem = self.forward_ssm(semantic)
            log["inputs"] = x
            log["semantic_true"] = torch.argmax(semantic, dim=1).unsqueeze(1) * 8/255
            log["semantic_reconstructed"] = torch.argmax(xrec_sem, dim=1).unsqueeze(1) * 8/255
            # For visualization
            log["binary_masked_inputs"] = x * mask_out_sem["binary_map"]
            log["scored_inputs"] = build_score_image(x, mask_out_sem["score_map"], scaler=0.7)
            # If you want the index map:
            if info_sem is not None and len(info_sem) > 2:
                ordered_vqidk = torch.full(
                    (x.shape[0], 16*32), fill_value=-1, dtype=torch.int64, device=x.device
                )
                ordered_vqidk.scatter_(1, mask_out_sem["sample_index"], info_sem[2])
                ordered_vqidk = ordered_vqidk.view(-1, 16, 32).unsqueeze(1)
                log["ordered_vqidk"] = ordered_vqidk

        elif self.mode == "img":
            # forward_img
            xrec_img, _, info_img, mask_out = self.forward_img(x, semantic)
            log["inputs"] = x
            log["reconstructions"] = xrec_img
            log["semantic_true"] = torch.argmax(semantic, dim=1).unsqueeze(1) * 8/255
            log["binary_masked_inputs"] = x * mask_out["binary_map"]
            log["scored_inputs"] = build_score_image(x, mask_out["score_map"], scaler=0.7)
            if info_img is not None and len(info_img) > 2:
                ordered_vqidk = torch.full(
                    (x.shape[0], 16*32), fill_value=-1, dtype=torch.int64, device=x.device
                )
                ordered_vqidk.scatter_(1, mask_out["sample_index"], info_img[2])
                ordered_vqidk = ordered_vqidk.view(-1, 16, 32).unsqueeze(1)
                log["ordered_vqidk"] = ordered_vqidk

        else:  # "all"
            # forward_all
            (xrec_img, _, info_img, 
             xrec_sem, _, info_sem, mask_out_img, mask_out_sem
            ) = self.forward_all(x, semantic)

            log["inputs"] = x
            log["reconstructions"] = xrec_img
            log["semantic_true"] = torch.argmax(semantic, dim=1).unsqueeze(1) * 8/255
            log["semantic_reconstructed"] = torch.argmax(xrec_sem, dim=1).unsqueeze(1) * 8/255
            log["binary_masked_inputs"] = x * mask_out_img["binary_map"]
            log["semantic_binary_masked_inputs"] = x * mask_out_sem["binary_map"]
            log["scored_inputs"] = build_score_image(x, mask_out_img["score_map"], scaler=0.7)

            if info_img is not None and len(info_img) > 2:
                ordered_vqidk_img = torch.full(
                    (x.shape[0], 16*32), fill_value=-1, dtype=torch.int64, device=x.device
                )
                ordered_vqidk_img.scatter_(1, mask_out_img["sample_index"], info_img[2])
                ordered_vqidk_img = ordered_vqidk_img.view(-1, 16, 32).unsqueeze(1)
                log["ordered_vqidk"] = ordered_vqidk_img

            if info_sem is not None and len(info_sem) > 2:
                ordered_vqidk_sem = torch.full(
                    (x.shape[0], 16*32), fill_value=-1, dtype=torch.int64, device=x.device
                )
                ordered_vqidk_sem.scatter_(1, mask_out_sem["sample_index"], info_sem[2])
                ordered_vqidk_sem = ordered_vqidk_sem.view(-1, 16, 32).unsqueeze(1)
                log["semantic_ordered_vqidk"] = ordered_vqidk_sem

        return log
