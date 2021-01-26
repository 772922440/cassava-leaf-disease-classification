
import torchvision.models as models
import torch.nn as nn
import timm

class TimmBackbone(nn.Module):
    def __init__(self, model_name= 'mobilenetv3_large_100', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # n_features = self.model.fc.in_features
        self.model.reset_classifier(target_size)

    def forward(self, x):
        x = self.model(x)
        return x