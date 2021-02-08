from .resnet import *

# mobilenetv3_large_100
# resnet18
# resnext50_32x4d
# tf_efficientnet_b3
def get_backbone(backbone, config):
    if config.forward_features:
        backbone = TimmBackboneFeature(model_name=backbone, target_size=config.target_size)
    elif config.addclf:
        backbone = TimmBackboneAddClf(model_name=backbone, target_size=config.target_size)
    else:
        backbone = TimmBackbone(model_name=backbone, target_size=config.target_size)
    return backbone