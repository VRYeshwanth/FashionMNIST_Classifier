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

class CNN(nn.Module):
    def __init__(self, dropout=0.0, batch_norm=False):
        super().__init__()

        conv_layers = [
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=3, stride=1, padding=1
            )
        ]

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(32))
        conv_layers.append(nn.ReLU())

        conv_layers.append(
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            )
        )
        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(64))
        conv_layers.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.conv_block = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)

        return x