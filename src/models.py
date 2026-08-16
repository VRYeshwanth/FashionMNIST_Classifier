from torch import nn

# Baseline CNN Model
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=64*14*14, out_features=128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)

        return self.classifier(x)