from .segmenter_detris import DETRIS
from loguru import logger


def build_segmenter_detris(args):
    model = DETRIS(args)
    backbone, head, neck, decoder, proj, fix = [], [], [], [], [], []
    for k, v in model.named_parameters():
        if (k.startswith('txt_backbone') and 'positional_embedding' not in k or 'dinov2' in k) and v.requires_grad:
            backbone.append(v)
        elif v.requires_grad:
            head.append(v)
            if 'neck' in k:
                neck.append(v)
            elif 'decoder' in k:
                decoder.append(v)
            elif 'proj' in k:
                proj.append(v)
        else:
            fix.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [
        {'params': backbone, 'initial_lr': args.lr_multi * args.base_lr},
        {'params': head,     'initial_lr': args.base_lr},
    ]
    return model, param_list
