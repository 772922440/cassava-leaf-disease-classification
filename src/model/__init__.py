from .resnet import *

# mobilenetv3_large_100
# resnet18
# resnext50_32x4d
# tf_efficientnet_b3
def get_backbone(backbone, config):
    backbone = TimmBackbone(model_name=backbone, target_size=config.target_size)
    return backbone