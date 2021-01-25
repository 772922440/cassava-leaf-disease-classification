from .resnet import *

def get_backbone(backbone, config):
    if backbone=='resnet18':
        backbone = CustomResNext18(target_size=config.target_size)
    elif backbone=='resnext50_32x4d':
        backbone = CustomResNext50(target_size=config.target_size)
    elif backbone=='mobilenetv3_large_100':
        backbone = MobileNetV3(target_size=config.target_size)
    else:
        backbone = MobileNetV3(model_name=backbone, target_size=config.target_size)
    return backbone