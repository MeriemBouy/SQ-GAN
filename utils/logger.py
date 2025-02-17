# for pytorch_lightning ModelCheckpoint, Callback, LearningRateMonitor, ... modules
import os
import wandb
from omegaconf import OmegaConf
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, argv_content=None):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
    
        self.argv_content = argv_content

    # 在pretrain例程开始时调用。
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
            
            with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
                f.write(str(self.argv_content))
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class CaptionImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp, type="wandb"):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.type = type  # wandb or tensorboard
        self.max_values_clamp = torch.tensor([2.24, 2.42, 2.64]).view(1, 3, 1, 1)
        self.min_values_clamp = torch.tensor([-2.11, -2.03, -1.80]).view(1, 3, 1, 1)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *kwargs):
        self.log_img(pl_module, batch, batch_idx, split="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *kwargs):
        self.log_img(pl_module, batch, batch_idx, split="val")
    
    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            if k == "ordered_vqidk" or k == "semantic_ordered_vqidk":
                pass
            else:
                grid = torchvision.utils.make_grid(images[k], normalize=True)
                grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids, commit=False)
    
    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=True)
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
    
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (batch_idx % self.batch_freq == 0) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and (self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)
            
            if "groundtruth_captions" in images:
                del images['groundtruth_captions']
            
            if "dest_captions" in images:
                del images['dest_captions']
            
            if "sample_captions" in images:
                del images['sample_captions']

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp and k in ["inputs", "reconstructions", "binary_masked_inputs", "semantic_binary_masked_inputs"]:
                        images[k] = torch.max(torch.min(images[k], self.max_values_clamp), self.min_values_clamp) #torch.clamp(images[k], -1., 1.)
            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            
            if is_train:
                pl_module.train()
    
    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        inv_normalize = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        ])
        
        for k in images:
            ###########
            ### Create the image that shows which are the codeword used
            ###########
            # This takes some times and it can be removed
            if k == "ordered_vqidk" or k == "semantic_ordered_vqidk":
                B = images[k].shape[0]  # Number of images in the batch
                ordered_vqidk = images[k]

                # Determine the number of rows and columns
                rows = 2 if B > 4 else 1
                cols = min(B, 4)

                # Initialize a matplotlib figure with the correct number of rows and columns
                fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4))

                # Flatten axs array for easy iteration if there are two rows
                axs = np.array(axs).reshape(-1) if B > 1 else [axs]

                # Define a colormap from blue to red
                cmap = cm.get_cmap('coolwarm')

                # Normalize the values between 0 and 1024 for color mapping
                norm = plt.Normalize(0, 1024)

                # Iterate over each image in the batch
                for idx, ax in enumerate(axs):
                    if idx < B:
                        ax.set_xlim(0, 32)
                        ax.set_ylim(16, 0)  # Inverted y-axis for proper display
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # Extract the single image of shape (16, 32)
                        image = ordered_vqidk[idx, 0]

                        # Iterate over each element in the image
                        for i in range(16):
                            for j in range(32):
                                value = int(image[i, j])
                                if value == -1:
                                    # Draw a gray box
                                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
                                    ax.add_patch(rect)
                                else:
                                    # Map the value to a color
                                    color = cmap(norm(value))
                                    # Draw a box with the number inside
                                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
                                    ax.add_patch(rect)
                                    ax.text(j + 0.5, i + 0.5, str(value), ha='center', va='center', fontsize=8)
                    else:
                        ax.axis('off')  # Turn off axes for empty plots

                # Save the resulting image
                plt.tight_layout()
                filename = "Step_{:06}-Epoch_{:03}-Batch_{:06}-{}.png".format(global_step, current_epoch, batch_idx, k)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                plt.savefig(path)
                plt.close(fig)  # Close the figure to free memory# Close the figure to free memory
            
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=False)
                if k in ["inputs", "reconstructions", "binary_masked_inputs", "semantic_binary_masked_inputs"]:
                    grid = inv_normalize(grid)
                grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid*255).astype(np.uint8)
                filename = "Step_{:06}-Epoch_{:03}-Batch_{:06}-{}.png".format(global_step,current_epoch,batch_idx,k)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)


if __name__ == "__main__":
    pass