from shutil import RegistryError
from .resnet import CustomResNext50, CustomResNext18

REGISTRY = {}
REGISTRY['resnext50_32x4d'] = CustomResNext50
REGISTRY['resnet18'] = CustomResNext18