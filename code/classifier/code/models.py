import torch.nn as nn
import torchvision.models


NUM_CLASSES = 60


class ResNet50(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet_sup_in1k",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        freeze_bn_statistics: bool = True
    ):
        super().__init__()
        if model_name == "resnet_sup_in1k":
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            self.n_outputs = 2048
            self.network.fc = nn.Linear(self.n_outputs, num_classes)
        self.freeze_bn_statistics = freeze_bn_statistics

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_statistics:
            self.freeze_bn()
        return self

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        return self.network(x)
