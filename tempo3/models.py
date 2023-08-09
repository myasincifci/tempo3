from torch import nn
from torchvision.models import ResNet34_Weights, resnet34
from lightly.models.modules import BarlowTwinsProjectionHead

class Tempo(nn.Module):
    """
    Tempo module consisting of a ResNet-34 feature extractor and a projection 
    head. Used for self supervised training.
    """

    def __init__(self, pretrain: bool = True, embedding_dim: int = 1024) -> None:
        super(Tempo, self).__init__()

        if pretrain:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BarlowTwinsProjectionHead(
            512, embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


if __name__ == "__main__":
    pass
