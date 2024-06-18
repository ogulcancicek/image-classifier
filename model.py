import timm
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, num_classes):
        super(ExampleModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        network_outsize = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(network_outsize, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output