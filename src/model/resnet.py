
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import timm

class TimmBackbone(nn.Module):
    def __init__(self, model_name= 'mobilenetv3_large_100', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(target_size)

    def forward(self, x):
        x = self.model(x)
        return x

# only for efficientnet
class TimmBackboneFeature(nn.Module):
    def __init__(self, model_name= 'tf_efficientnet_b3', pretrained=True, target_size = 5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(target_size)

    def forward(self, x):
        x = self.model.forward_features(x)
        xx = self.model.global_pool(x)
        if self.model.drop_rate > 0.:
            xx = F.dropout(xx, p=self.model.drop_rate, training=self.model.training)
        return self.model.classifier(xx), xx



class TimmBackboneAddClf(nn.Module):
    def __init__(self, model_name= 'mobilenetv3_large_100', pretrained=True, target_size = 5):
        super().__init__()

        self.dim_m1 = 512
        self.dim_m2 = 256


        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.clf = nn.Sequential(
            nn.Linear(1000, self.dim_m1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_m1),
            nn.Dropout(p=0.3),
            
            nn.Linear(self.dim_m1, self.dim_m2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_m2),
            nn.Dropout(p=0.3),

            nn.Linear(self.dim_m2, target_size),
            
            )

    def forward(self, x):
        x = self.model(x)
        x = self.clf(x)
        return x