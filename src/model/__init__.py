from shutil import RegistryError
from .resnet import *

def get_backbone(backbone, config):
    if backbone=='resnet18':
        backbone = CustomResNext18(target_size=config.target_size)
    elif backbone=='resnext50_32x4d':
        backbone = CustomResNext50(target_size=config.target_size)
    else:
        raise "backbone error"
    return backbone