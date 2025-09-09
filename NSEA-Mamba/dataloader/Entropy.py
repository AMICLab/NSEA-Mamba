import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import pytorch_lightning as pl


class Entropy(nn.Module):
    def __init__(self, patch_size):
        super(Entropy, self).__init__()
        self.psize = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int, h_num: int,
                w_num: int) -> torch.Tensor:
        epsilon = 1e-40
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, 1, h_num, w_num))  # 保持通道数为1
        return entropy

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = inputs.shape
        h_num = int(height // self.psize)
        w_num = int(width // self.psize)

        unfolded_images = self.unfold(inputs)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = unfolded_images.reshape(unfolded_images.shape[0] * h_num * w_num, -1)

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size, h_num=h_num, w_num=w_num)
        return entropy


class DynamicEntropyModel(pl.LightningModule):
    def __init__(self):
        super(DynamicEntropyModel, self).__init__()
        self.entropy_calculators = nn.ModuleDict({
            'p4': Entropy(4),
            'p2': Entropy(2),
            'p1': Entropy(1)
        })

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        entropy_image = torch.zeros_like(x)

        for i in range(0, height, 4):
            for j in range(0, width, 4):
                patch = x[:, :, i:i + 4, j:j + 4]
                if patch.shape[-1] < 4 or patch.shape[-2] < 4:
                    continue

                # 计算6x6尺度的熵
                entropy_patch_6 = self.entropy_calculators['p4'](patch)

                # 根据6x6尺度的熵值决定使用尺度
                if entropy_patch_6.max() > 0.3:  # 假设阈值为0.3
                    entropy_patch = self.entropy_calculators['p2'](patch)
                    if entropy_patch.max() > 0.5:  # 假设阈值为0.5
                        entropy_patch = self.entropy_calculators['p1'](patch)
                else:
                    entropy_patch = entropy_patch_6

                # 确保熵补丁的尺寸与目标区域匹配
                if entropy_patch.shape[-2:] != (4, 4):
                    entropy_patch = F.interpolate(entropy_patch, size=(4, 4), mode='bilinear', align_corners=True)

                entropy_image[:, :, i:i + 4, j:j + 4] = entropy_patch

        return entropy_image
