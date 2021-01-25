
import torchvision.models as models
import torch.nn as nn
import timm

class CustomResNext50(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.model.fc.in_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomResNext18(nn.Module):
    def __init__(self, model_name= 'resnet18', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.model.fc.in_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x

class MobileNetV3(nn.Module):
    def __init__(self, model_name= 'mobilenetv3_large_100', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # n_features = self.model.fc.in_features
        self.model.reset_classifier(target_size)

    def forward(self, x):
        x = self.model(x)
        return x