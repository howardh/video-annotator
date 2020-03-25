import torch
import torchvision

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)
        self.seq = torch.nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.base.avgpool
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Linear(in_features=256,out_features=1)
        )
        self.coordinate = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Linear(in_features=256,out_features=2),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.seq(x)
        x = x.squeeze()
        visible = self.classifier(x)
        coord = self.coordinate(x)
        return coord, visible

