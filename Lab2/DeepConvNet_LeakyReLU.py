import torch.nn  as nn

class DeepConvNet_LeakyReLU(nn.Module):

    def __init__(self):
        super(DeepConvNet_LeakyReLU, self).__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size= (1, 5)),
            nn.Conv2d(25, 25, kernel_size= (2, 1)),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(25, 50, kernel_size= (1, 5)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(50, 100, kernel_size= (1, 5)),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),
            nn.Conv2d(100, 200, kernel_size= (1, 5)),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout(),

            nn.Conv2d(200, 330, kernel_size= (1, 5)),
            nn.BatchNorm2d(330),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (1, 2)),
            nn.Dropout()
        )

        self.fc = nn.Sequential(
            nn.Linear(6270, 2)
        )

    def forward(self, x):

        x = self.conv2d(x)
        x = x.flatten(1)
        return self.fc(x)
