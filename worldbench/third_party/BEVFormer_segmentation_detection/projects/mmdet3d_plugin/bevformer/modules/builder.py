from mmcv.utils import Registry, build_from_cfg

SEG_ENCODER = Registry('seg_encoder')

def build_seg_encoder(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, SEG_ENCODER, default_args)
