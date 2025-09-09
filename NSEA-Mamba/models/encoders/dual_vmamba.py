import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock

logger = get_logger()

class ConcatMambaFusionBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio, d_state):
        super(ConcatMambaFusionBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_g, x_e):
        x_g = self.layer1(x_g)
        x_g = torch.relu(x_g)
        x_g = self.layer2(x_g)

        x_e = self.layer1(x_e)
        x_e = torch.relu(x_e)
        x_e = self.layer2(x_e)

        return x_g, x_e

class PETTransformer(nn.Module):
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 9, 2],  
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[128, 128],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_g, x_e):
        """
        x_g: B x C x H x W
        """
        outs_g = self.vssm(x_g)  # B x C x H x W
        outs_e = self.vssm(x_e)  # B x C x H x W

        outs_fused = []

        for i in range(4):
            if self.ape:
                out_g = self.absolute_pos_embed[i].to(outs_g[i].device) + outs_g[i]
                out_e = self.absolute_pos_embed_x[i].to(outs_e[i].device) + outs_e[i]
            else:
                out_g = outs_g[i]
                out_e = outs_e[i]

            cross_g, cross_e = self.cross_mamba[i](out_g.permute(0, 2, 3, 1).contiguous(),
                                                       out_e.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
            x_g, x_e = self.channel_attn_mamba[i](cross_g, cross_e)
            x_fuse = x_g.permute(0, 3, 1, 2).contiguous()

            outs_fused.append(x_fuse)

        return outs_fused

    def forward(self, x_g, x_e):
        # 调用 forward_features 方法并返回结果
        return self.forward_features(x_g, x_e)

class vssm_tiny(PETTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            # pretrained 参数被删除或设为 None
            pretrained=None
        )
class vssm_small(PETTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained=None,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(PETTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained=None,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
