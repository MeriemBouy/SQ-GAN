import os
import sys
import torch
import pandas as pd
from torchmetrics import PeakSignalNoiseRatio, MeanSquaredError, MultiScaleStructuralSimilarityIndexMeasure
from modules.losses.lpips import LPIPS
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate.metric_functions as mfunc  # Import the custom metric functions
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

internimage_path = "/kaggle/working/InternImage"


# Function to load the model
def load_model(config, checkpoint_path=None, ckpt_path_parts=None):
    model = instantiate_from_config(config.model)
    
    if checkpoint_path:
        # Load from a single checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded model from single checkpoint: {checkpoint_path}")
    elif ckpt_path_parts:
        # Load from multiple submodel checkpoints
        for ckpt_file in os.listdir(ckpt_path_parts):
            if not ckpt_file.endswith(".ckpt"):
                continue
            attr_name = ckpt_file.split(".")[0]  # e.g., "encoder_semantic"
            full_path = os.path.join(ckpt_path_parts, ckpt_file)
            if hasattr(model, attr_name):
                submodel = getattr(model, attr_name)
                sd = torch.load(full_path, map_location='cpu')
                submodel.load_state_dict(sd, strict=False)
                print(f"Loaded submodule '{attr_name}' from '{ckpt_file}'.")
            else:
                print(f"Model does not have attribute '{attr_name}'. Skipping.")
    else:
        raise ValueError("Either 'checkpoint_path' or 'ckpt_path_parts' must be provided.")
    
    model.eval()
    return model

# Function to save the images during generation
def save_images(preds, original, semantic, rec_sem, mask_output, mask_output_sem, image_folder, img_name, clamp=True, save_original=False):
    os.makedirs(image_folder, exist_ok=True)

    max_values_clamp = torch.tensor([2.24, 2.42, 2.64]).view(1, 3, 1, 1).to('cuda:0')
    min_values_clamp = torch.tensor([-2.11, -2.03, -1.80]).view(1, 3, 1, 1).to('cuda:0')
    
    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    ])
    
    def process_and_save(img_tensor, folder_name, clamp, save_original_flag=None):
        if save_original_flag:
            save_path = os.path.join(image_folder.split("img_")[0], folder_name)
        else:
            save_path = os.path.join(image_folder, folder_name)
        os.makedirs(save_path, exist_ok=True)
        if clamp:
            img_tensor = torch.max(torch.min(img_tensor, max_values_clamp), min_values_clamp)
            img_tensor = inv_normalize(img_tensor)
        try:
            img = img_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        except:
            img = img_tensor.squeeze().cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_path, img_name[0]))

    if save_original:
        process_and_save(original, "original_image", True, save_original)
        process_and_save(semantic / 255 * 7, "original_semantic", False, save_original)
        
    process_and_save(preds, "reconstructed_image", True)
    process_and_save(rec_sem / 255 * 7, "reconstructed_semantic", False)
    process_and_save(original * mask_output, "masked_image", True)
    process_and_save(original * mask_output_sem, "masked_semantic", True)


# Function to run the data generation
def generate_data(model, config, img_masking_fraction, ssm_masking_fraction, base_save_path, save_original=False):
    data_module = instantiate_from_config(config.data)
    data_module.setup('test')
    
    test_loader = data_module.test_dataloader()

    save_path = f"{base_save_path}/img_{img_masking_fraction:.2f}_ssm_{ssm_masking_fraction:.2f}/"
    os.makedirs(save_path, exist_ok=True)
    
    model.masker.topk_ratio_prob = "fixed"
    model.masker_semantic.topk_ratio_prob = "fixed"
    model.masker.topk_ratio_range = [img_masking_fraction, img_masking_fraction+1]
    model.masker_semantic.topk_ratio_range = [ssm_masking_fraction, ssm_masking_fraction+1]
    
    model.eval()
    for batch in tqdm(test_loader):
        x = batch['image'].to(model.device)
        semantic = batch['semantic'].to(model.device)
        img_name = batch['filename']

        # reconstruct the img and the ssm
        if config.model.params.mode == "all":
            xrec_img, emb_loss_img, info_img, xrec_sem, emb_loss_sem, info_sem, mask_out_img, mask_out_sem = model.forward_all(x, semantic)
        else:
            raise ValueError(f"Unknown mode {config.model.params.mode}, must be 'all'.")    
        
        
        save_images(
            preds=xrec_img,
            original=x,
            semantic=torch.argmax(semantic, dim=1, keepdim=True),
            rec_sem=torch.argmax(xrec_sem, dim=1, keepdim=True),
            mask_output=mask_out_img["binary_map"],
            mask_output_sem=mask_out_sem["binary_map"],
            image_folder=save_path,
            img_name=img_name,
            save_original=save_original
        )


# Function to calculate metrics from saved images
def evaluate_metrics_from_images(image_folder, img_masking_fraction, ssm_masking_fraction, base_results_path, internimage_path=internimage_path):
    # define paths
    original_image_folder = os.path.join(image_folder.split("img_")[0], "original_image")
    reconstructed_image_folder = os.path.join(image_folder, "reconstructed_image")
    original_semantic_folder = os.path.join(image_folder.split("img_")[0], "original_semantic/")
    reconstructed_semantic_folder = os.path.join(image_folder, "reconstructed_semantic/")
    intermimage_semantic_path = os.path.join(image_folder, "sem_generated/")
    # define metrics
    mse_metric = MeanSquaredError()
    psnr_metric = PeakSignalNoiseRatio()
    lpips_metric = LPIPS()  # Choose 'alex' or 'vgg' based on your needs
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure()

    metrics = []
    print(f"Evaluating metrics for mask_percentage: {img_masking_fraction}, semantic_percentage: {ssm_masking_fraction}")
    for img_name in tqdm(os.listdir(reconstructed_image_folder), desc="Evaluating metrics"):
        original_image_path = os.path.join(original_image_folder, img_name)
        reconstructed_image_path = os.path.join(reconstructed_image_folder, img_name)

        # Skip if original image doesn't exist
        if not os.path.exists(original_image_path):
            print(f"Original image not found: {original_image_path}. Skipping.")
            continue

        original_image = Image.open(original_image_path).convert("RGB")
        reconstructed_image = Image.open(reconstructed_image_path).convert("RGB")
        
        original_tensor = transforms.ToTensor()(original_image).unsqueeze(0)
        reconstructed_tensor = transforms.ToTensor()(reconstructed_image).unsqueeze(0)

        mse = mse_metric(reconstructed_tensor, original_tensor).item() / torch.var(original_tensor).item()
        psnr = psnr_metric(reconstructed_tensor, original_tensor).item()
        lpips = lpips_metric(reconstructed_tensor, original_tensor).item()
        msssim = msssim_metric(reconstructed_tensor, original_tensor).item()

        # Calculate bits per pixel (bpp)
        bpp_img = 10 * int(512 * img_masking_fraction) / (original_tensor.shape[2] * original_tensor.shape[3]) + 512 / (original_tensor.shape[2] * original_tensor.shape[3])
        bpp_sem = 10 * int(512 * ssm_masking_fraction) / (original_tensor.shape[2] * original_tensor.shape[3]) + 512 / (original_tensor.shape[2] * original_tensor.shape[3])

        metrics.append({
            'mse': mse,
            'psnr': psnr,
            'lpips': lpips,
            'msssim': msssim,
            'bpp_img': bpp_img,
            'bpp_sem': bpp_sem
        })

    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics]),
        'psnr': np.mean([m['psnr'] for m in metrics]),
        'lpips': np.mean([m['lpips'] for m in metrics]),
        'msssim': np.mean([m['msssim'] for m in metrics]),
        'bpp_img': np.mean([m['bpp_img'] for m in metrics]),
        'bpp_sem': np.mean([m['bpp_sem'] for m in metrics])
    }
    
    #### Evaluate the generated SSM
    relative_path = os.path.join(os.getcwd(), reconstructed_image_folder)
    relative_path_sem_generated = os.path.join(os.path.dirname(relative_path), "sem_generated")
    # Re-run the semantic evaluation if needed
    command = f"cd {internimage_path} && python3 segmentation/image_demo.py {relative_path} segmentation/configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py segmentation/checkpoint_dir/seg/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth --out {relative_path_sem_generated}"
    mfunc.run_command_with_conda_env("internimage", command)
    
    # evaluate mIoU
    true_SSM = mfunc.get_semantic_maps(original_semantic_folder)
    gen_SSM = mfunc.get_semantic_maps(reconstructed_semantic_folder)
    internimage_SSM = mfunc.get_semantic_maps(intermimage_semantic_path, interimage=True)
    miou = mfunc.calculate_mIoU(true_SSM, gen_SSM)
    miou_internimage = mfunc.calculate_mIoU(true_SSM, internimage_SSM)
    
    # evaluate FIF
    fid = mfunc.calculate_FID(original_image_folder, reconstructed_image_folder)

    # aggregate these metrics to the previous one and save
    avg_metrics.update({
        'mIoU': miou,
        'FID': fid,
        'mIoU_internimage': miou_internimage
    })

    df_metrics = pd.DataFrame([avg_metrics])
    df_metrics.to_csv(os.path.join(image_folder, "metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(image_folder, 'metrics.csv')}")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test script for SQGAN model.")
    parser.add_argument("--base", type=str, required=True, help="Path to the base configuration file.")
    
    # Create a mutually exclusive group for checkpoints
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume_from_checkpoint", type=str, help="Path to a single checkpoint to resume from.")
    group.add_argument("--ckpt_path_parts", type=str, help="Path to the directory containing submodel checkpoints.")
    
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU ids to use.")
    # Add other arguments as needed, e.g., --mode, etc.
    group.add_argument("--result_path",  default="Results", type=str, help="Path where to store the results")
    parser.add_argument("--mode", type=str, default="all", choices=["ssm", "img", "all"], help="speciy which training step you want to perform",)
    
    
    args = parser.parse_args()
    
    config_path = args.base
    checkpoint_path = args.resume_from_checkpoint
    ckpt_path_parts = args.ckpt_path_parts
    base_save_path = args.result_path #"/home/pezone/MaskedVectorQuantization/Results"  # Adjust as needed
    mode = args.mode
    
    # Load the configuration
    config = OmegaConf.load(config_path)
    # Inject the mode parameter into the configuration
    config.model.params.mode = mode
    # Load the model
    model = load_model(config, checkpoint_path=checkpoint_path, ckpt_path_parts=ckpt_path_parts).to('cuda:0')
    
    # Define the masking fractions for the image and ssm pipelines
    img_masking_fractions = np.array([0.4, 0.7, 1.0])
    ssm_masking_fractions = np.array([0.4, 0.6, 1.0])

    first_run = True 
    # Generate data for all combinations of mask and semantic percentages
    for img_masking_fraction in img_masking_fractions:
        for ssm_masking_fraction in ssm_masking_fractions:
            generate_data(
                model=model,
                config=config,
                img_masking_fraction=img_masking_fraction,
                ssm_masking_fraction=ssm_masking_fraction,
                base_save_path=base_save_path,
                save_original=first_run  # Set to True if you want to save original images
            )
            first_run = False
            
    # Evaluate metrics from saved images for all combinations
    for img_masking_fraction in img_masking_fractions:
        for ssm_masking_fraction in ssm_masking_fractions:
            image_folder = f"{base_save_path}/img_{img_masking_fraction:.2f}_ssm_{ssm_masking_fraction:.2f}/"
            evaluate_metrics_from_images(
                image_folder=image_folder,
                img_masking_fraction=img_masking_fraction,
                ssm_masking_fraction=ssm_masking_fraction,
                base_results_path=base_save_path
            )
    
    
