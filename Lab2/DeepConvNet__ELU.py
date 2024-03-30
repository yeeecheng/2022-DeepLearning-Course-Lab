import torch.nn  as nn

class DeepConvNet_ELU(nn.Module):

    def __init__(self):
        super(DeepConvNet_ELU, self).__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size= (1, 5)),
            nn.Conv2d(25, 25, kernel_size= (2, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(25, 50, kernel_size= (1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(50, 100, kernel_size= (1, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(100, 200, kernel_size= (1, 5)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout()
        )

        self.fc = nn.Sequential(
            nn.Linear(8600, 2)
        )

    def forward(self, x):

        x = self.conv2d(x)
        x = x.flatten(1)
        return self.fc(x)
