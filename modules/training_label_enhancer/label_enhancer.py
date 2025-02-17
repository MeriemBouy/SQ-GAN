import torch
import numpy as np
import scipy.ndimage
import random


class LabelEnhancer:
    def __init__(self, ):
        pass
    
    def find_connected_components(self, mask):
        mask_np = mask.cpu().numpy().astype(np.uint8)
        labeled_array, num_features = scipy.ndimage.label(mask_np)
        return torch.tensor(labeled_array, device=mask.device), num_features

    def extract_sub_masks(self, labeled_array, num_features):
        sub_masks = []
        for i in range(1, num_features + 1):
            sub_mask = (labeled_array == i).float()
            sub_masks.append(sub_mask)
        return sub_masks

    def pick_random_non_overlapping_sub_masks(self, sub_masks, original_mask, mask_shape, device, max_sub_masks):
        chosen_sub_masks = []
        combined_mask = original_mask.clone()
        
        for i in range(len(sub_masks)):
            sub_mask = sub_masks[i]
            # check for overlapping
            if torch.all(combined_mask[sub_mask == 1] == 0):
                chosen_sub_masks.append(sub_mask)
                combined_mask[sub_mask == 1] = 1
                if len(chosen_sub_masks) >= max_sub_masks:
                    break
        return chosen_sub_masks

    def process_masks(self, masks, x, s):
        B, _, H, W = masks.shape
        all_sub_masks = []
        sub_masks_info = []

        # Extract all sub-masks from all the masks and store their batch index
        for i in range(B):
            mask = masks[i, 0]
            labeled_array, num_features = self.find_connected_components(mask)
            sub_masks = self.extract_sub_masks(labeled_array, num_features)
            all_sub_masks.extend(sub_masks)
            sub_masks_info.extend([(i, sub_mask) for sub_mask in sub_masks])

        # sort the sub-masks by size from largest to smallest
        sub_masks_info = sorted(sub_masks_info, key=lambda x: torch.sum(x[1]), reverse=True)
        
        
        # Now starts the process of enhancing
        result_masks = torch.zeros((B,1, H, W), device=masks.device)
        result_x = x.clone()
        result_s = s.clone()
        
        for i in range(B):
            # 90% of the case Enhance  
            if random.random() < 0.9:
                original_mask = masks[i, 0]
                max_sub_masks = np.random.randint(1, len(all_sub_masks)//2 + 2)  # Random number of sub-masks to pick
                chosen_sub_masks = self.pick_random_non_overlapping_sub_masks([sm[1] for sm in sub_masks_info], original_mask, (H, W), masks.device, max_sub_masks) # select the non-overlapping sub-masks
                
                # update the img/ssm by inserting the new objects
                new_mask = torch.zeros((H, W), device=masks.device)
                for sub_mask in chosen_sub_masks:
                    new_mask[sub_mask == 1] = 1
                    # Find the corresponding x_j and s_j and replace parts of x_i and s_i
                    for (j, sm) in sub_masks_info:
                        if torch.equal(sm, sub_mask):
                            result_x[i] = torch.where(sub_mask.unsqueeze(0) == 1, x[j], result_x[i])
                            result_s[i] = torch.where(sub_mask.unsqueeze(0) == 1, s[j], result_s[i])

                result_masks[i,0] = new_mask
            else:
                result_masks[i,0] = masks[i,0]

        return result_masks, result_x, result_s

