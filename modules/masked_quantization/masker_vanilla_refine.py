import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F


class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16

        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        return x * (1 + gamma) + beta
    
class ScorePredictor(nn.Module):
    def __init__(self, input_dim, label_nc):
        super(ScorePredictor, self).__init__()
        self.layer1 = SPADEGroupNorm(input_dim, label_nc)
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.layer2 = SPADEGroupNorm(input_dim, label_nc)
        self.conv2 = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, segmap):
        # Apply the first SPADEGroupNorm layer
        x = self.layer1(x, segmap)

        # Apply the first convolution and activation
        x = self.conv1(x)
        x = self.relu(x)

        # Apply the second SPADEGroupNorm layer
        x = self.layer2(x, segmap)

        # Apply the second convolution
        x = self.conv2(x)

        # Apply the sigmoid activation
        x = self.sigmoid(x)

        return x

# first predict scores, then norm features
# replace score net with a conv one 
class VanillaMasker(nn.Module):
    def __init__(self, 
                 topk_ratio_range,
                 topk_ratio_prob,
                 input_token_num,
                 input_dim,
                 patch_size,
                 score_pred_net_mode = "2layer",
                 codebook_dim = 32,
                 ratio_dim = 1,
                 num_heads=8,
                 subdivisions=5,
                 ):
        super().__init__()
        self.hw = int(input_token_num**0.5)
        self.ratio_dim = ratio_dim
        self.input_token_num = self.hw * self.hw*self.ratio_dim
        self.topk_ratio_range = topk_ratio_range
        self.topk_ratio_prob = topk_ratio_prob
        self.patch_size = patch_size
        self.subdivisions = subdivisions
         
        # cross attention here
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        
        if score_pred_net_mode == "2layer":
            self.score_pred_net = ScorePredictor(input_dim, 19)
        else:
            raise ValueError
        self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.mask = torch.from_numpy(np.zeros(self.input_token_num)).float()
        self.pre_projection = torch.nn.Linear(input_dim, codebook_dim, bias=False)
        
    def generate_values_in_range(self):
        # set the possible range values
        low, high = self.topk_ratio_range
        # deterministic value of the masking fraction (useful in evaluation)
        if self.topk_ratio_prob == 'fixed':
            return low
        # select at random the masking fraction depending on the distribution 
        elif self.topk_ratio_prob == 'uniform':
            return np.random.uniform(low, high)
        elif self.topk_ratio_prob == 'beta':
            alpha = 3.  # parameters to fit the distribution of the masked tokens depending on the cityscape dataset
            beta = 5.
            value = np.random.beta(alpha, beta)
            range_v = np.linspace(low, high, self.subdivisions)
            value = range_v[np.argmin(np.abs(range_v - value))]
            return np.clip(value, low, high)
        elif self.topk_ratio_prob == 'log-normal':
            mu=-0.95
            sigma=0.4
            value=np.random.lognormal(mean=mu, sigma=sigma)
            range_v=np.linspace(low, high, self.subdivisions)
            value=range_v[np.argmin(np.abs(range_v-value))]
            return np.clip(value, low, high)
        elif self.topk_ratio_prob == 'gaussian':
            mean = (high + low) / 2
            std_dev = (high - low) / 6  # Approximately 99.7% values will be within the range
            value = np.random.normal(mean, std_dev)
            # Ensure the value is within the specified range
            return np.clip(value, low, high)
        elif self.topk_ratio_prob == 'custom':
            assert low == high, "low and high must be equal for custom distribution"
            return low
        else:
            raise ValueError("Unsupported distribution. Choose 'uniform' or 'gaussian'.")

    def forward(self, image_features, semantic):
        ## dinamic topk setup
        topk_ratio = self.generate_values_in_range()
        sample_num = int(topk_ratio * self.input_token_num)
        unsampled_num = self.input_token_num - sample_num
        topk_num = int(topk_ratio * self.input_token_num)
        
        batch_size, channel, height, width = image_features.size()
        
        # evaluate the relevance scores
        pred_score = self.score_pred_net(image_features, semantic).view(batch_size, -1)
        pred_score_clone = pred_score.clone().detach()
        # sort the elements and select the first topk_num
        sort_score, sort_order = pred_score_clone.sort(descending=True, dim=1)
        sort_topk = sort_order[:, :topk_num]
        sort_topk_remain = sort_order[:, topk_num:]
        
        ## flatten for gathering
        image_features = rearrange(image_features, "B C H W -> B (H W) C")
        image_features = self.norm_feature(image_features)

        ## (only) sampled features multiply with score 
        # this is done to allow backpropagation
        image_features_sampled = image_features.gather(
            1, sort_topk[...,None].expand(-1, -1, channel)) * pred_score.gather(1, sort_topk).unsqueeze(-1)
        image_features_sampled = rearrange(self.pre_projection(image_features_sampled), "B N C -> B C N")

        # create the binary mask
        self.mask = self.mask.to(image_features_sampled.device)
        for i in range(batch_size):
            if i == 0:
                mask = self.mask.scatter(-1, sort_topk[i], 1.).view(self.hw, self.hw * self.ratio_dim).unsqueeze(0)
            else:
                mask_i = self.mask.scatter(-1, sort_topk[i], 1.).view(self.hw, self.hw * self.ratio_dim).unsqueeze(0)
                mask = torch.cat([mask, mask_i], dim=0)
        squeezed_mask = mask.view(batch_size, -1)  
        mask = F.interpolate(mask.float().unsqueeze(1), scale_factor=self.patch_size, mode="nearest")

        normed_score = pred_score_clone.sub(pred_score_clone.min()).div(max(pred_score_clone.max() - pred_score_clone.min(), 1e-5)).unsqueeze(-1)
        normed_score = F.interpolate(rearrange(normed_score, "b (h w) c -> b c h w", h=self.hw, w=self.hw * self.ratio_dim), scale_factor=self.patch_size, mode="nearest")

        return_dict = {
            "sample_features": image_features_sampled,
            "remain_features": None, 
            "sample_index": sort_topk,
            "remain_index": sort_topk_remain,
            "binary_map": mask,
            "score_map": normed_score,
            "squeezed_mask": squeezed_mask,
            "sort_score": sort_score[:, :topk_num],
            "sampled_length": image_features_sampled.size(-1),
        }
        
        return return_dict


if __name__ == "__main__":
    image_features = torch.randn(10, 256, 32, 32)
    masker = VanillaMasker(
        topk_ratio = 0.25,
        input_token_num = 1024,
        input_dim = 256,
        patch_size = 32,
        score_pred_net_mode = "2layer",
        codebook_dim = 256,
    )
    masker(image_features)