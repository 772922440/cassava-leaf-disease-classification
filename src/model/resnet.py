
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
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

# only for efficientnet
class TimmBackboneFeature(nn.Module):
    def __init__(self, model_name= 'tf_efficientnet_b3', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # n_features = self.model.fc.in_features
        self.model.reset_classifier(target_size)

    def forward(self, x):
        x = self.model.forward_features(x)
        xx = self.model.global_pool(x)
        if self.model.drop_rate > 0.:
            xx = F.dropout(xx, p=self.model.drop_rate, training=self.model.training)
        return self.model.classifier(xx), xx