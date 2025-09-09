import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Type, Optional, Sequence
from torch.nn.modules import module
from models.encoders.vmamba import CVSSDecoderBlock
import torch.utils.checkpoint as checkpoint
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """

        x = self.expand(x)  # B, H, W, 2C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)
        return x

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保 x 的形状是 (B, C, H, W)
        if x.dim() == 4 and x.shape[1] != self.dim:
            x = x.permute(0, 3, 1, 2).contiguous()  # 将 (B, H, W, C) 转换为 (B, C, H, W)

        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        # x = rearrange(x, pattern="b h w c -> b c h w")
        return x

class FLKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 4, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),  # 添加 ReLU 激活函数
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, 1, kernel_size=1)  # 单通道输出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保 x 的形状是 (B, C, H, W)
        if x.dim() == 4 and x.shape[1] != self.dim:
            x = x.permute(0, 3, 1, 2).contiguous()  # 将 (B, H, W, C) 转换为 (B, C, H, W)

        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        x = self.out(x)  # 应用输出层
        return x  # 返回单通道图像

class Mamba_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, dt_rank="auto",
                 d_state=4, ssm_ratio=2.0, attn_drop_rate=0.,
                 drop_rate=0.0, mlp_ratio=4.0,
                 drop_path=0.1, norm_layer=nn.LayerNorm, upsample=None,
                 shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
            for i in range(depth)])

        if upsample is not None:
            self.upsample = LKPE(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class MambaDecoder(nn.Module):
    def __init__(self,
                 img_size=[180, 180],
                 in_channels=[96, 192, 384, 768],
                 dropout_ratio=0.1,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                    input_resolution=(
                                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                    depth=depths[(self.num_layers - 1 - i_layer)],
                                    mlp_ratio=self.mlp_ratio,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                                    norm_layer=norm_layer,
                                    upsample=LKPE if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
            self.norm_ds = nn.ModuleList([norm_layer(embed_dim * 2 ** (self.num_layers - 2 - i_layer)) for i_layer in
                                          range(self.num_layers - 1)])
            self.output_ds = nn.ModuleList([nn.Conv2d(in_channels=embed_dim * 2 ** (self.num_layers - 2 - i_layer),
                                                      out_channels=1, kernel_size=1, bias=False) for
                                            i_layer in range(self.num_layers - 1)])

        self.up = FLKPE(dim=embed_dim, dim_scale=4, norm_layer=norm_layer)

    def forward_up_features(self, inputs):  # B, C, H, W
        if not self.deep_supervision:
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                else:
                    # interpolate y to input size (only pst900 dataset needs)
                    B, C, H, W = inputs[3 - inx].shape
                    y = F.interpolate(y.permute(0, 3, 1, 2).contiguous(), size=(H, W), mode='bilinear',
                                      align_corners=False).permute(0, 2, 3, 1).contiguous()

                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)

            x = self.norm_up(y)

            return x
        else:
            # if deep supervision
            x_upsample = []
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                    x_upsample.append(self.norm_ds[inx](y))
                else:
                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
                    if inx != self.num_layers - 1:
                        x_upsample.append((self.norm_ds[inx](y)))

            x = self.norm_up(y)

            return x, x_upsample

    def forward(self, inputs):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs)  # B, H, W, C
            x_last = self.up_x4(x)  # 只传递一个参数 x
            return x_last
        else:
            x, x_upsample = self.forward_up_features(inputs)
            x_last = self.up_x4(x)  # 只传递一个参数 x
            x_output_0 = self.output_ds[0](
                F.interpolate(x_upsample[0].permute(0, 3, 1, 2).contiguous(), scale_factor=16, mode='bilinear',
                              align_corners=False))
            x_output_1 = self.output_ds[1](
                F.interpolate(x_upsample[1].permute(0, 3, 1, 2).contiguous(), scale_factor=8, mode='bilinear',
                              align_corners=False))
            x_output_2 = self.output_ds[2](
                F.interpolate(x_upsample[2].permute(0, 3, 1, 2).contiguous(), scale_factor=4, mode='bilinear',
                              align_corners=False))
            return x_last, x_output_0, x_output_1, x_output_2

    def up_x4(self, x):
        x = self.up(x)  # 直接调用 FLKPE 实例
        return x
