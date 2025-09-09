import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_func import init_weight
from functools import partial
from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.MSELoss(), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.channels = [96, 192, 384,768]  # 默认通道数

        # 根据配置文件选择骨干网络
        self.select_backbone(cfg)

        self.aux_head = None

        if cfg.decoder == 'MambaDecoder':
            logger.info('Using Mamba Decoder')
            from .decoders.MambaDecoder import MambaDecoder
            self.decode_head = MambaDecoder(img_size=[cfg.image_height, cfg.image_width],
                                            in_channels=self.channels,
                                            embed_dim=self.channels[0],
                                            deep_supervision=False)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg)

    def select_backbone(self, cfg):
        if cfg and 'backbone' in cfg:
            backbone_type = cfg['backbone']
            if backbone_type == 'sigma_tiny':
                logger.info('Using backbone: V-MAMBA (Tiny)')
                self.channels = [96, 192, 384, 768]
                from .encoders.dual_vmamba import vssm_tiny as backbone
                self.backbone = backbone()
            elif backbone_type == 'sigma_small':
                logger.info('Using backbone: V-MAMBA (Small)')
                self.channels = [96, 192, 384, 768]
                from .encoders.dual_vmamba import vssm_small as backbone
                self.backbone = backbone()
            elif backbone_type == 'sigma_base':
                logger.info('Using backbone: V-MAMBA (Base)')
                self.channels = [128, 256, 512, 1024]
                from .encoders.dual_vmamba import vssm_base as backbone
                self.backbone = backbone()
            else:
                logger.error(f'Unknown backbone type: {backbone_type}')
                raise ValueError(f'Unknown backbone type: {backbone_type}')
        else:
            raise ValueError("Configuration must include 'backbone' key.")

    def init_weights(self, cfg):
        # 初始化权重，不加载预训练模型
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    def encode_decode(self, x_g, x_e):
        """Encode images with backbone and decode into a feature map."""
        orisize = x_g.shape[2:]  # 保存原始图像的大小
        x = self.backbone(x_g, x_e)
        out = self.decode_head.forward(x)
        return out

    def forward(self, x_g, x_e, label=None):
        out = self.encode_decode(x_g, x_e)
        return out

    def flops(self, shape=(1, 128, 128)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, 1, *shape), device=next(model.parameters()).device),
                 torch.randn((1, 1, *shape), device=next(model.parameters()).device))
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input)

        del model, input
        return f"params {params} GFLOPs {sum(Gflops.values())}"

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops
