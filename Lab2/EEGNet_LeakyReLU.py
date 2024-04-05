import torch.nn as nn

class EEGNet_LeakyReLU(nn.Module):
    def __init__(self):
        super(EEGNet_LeakyReLU, self).__init__()

        # input [1, 2, 750]
        self.first_conv2d = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 51), stride= (1, 1), padding=(0, 25), bias= False),
            nn.BatchNorm2d(8)
        )
        # [64, 2, 750]
        self.depthwise_conv2d = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size= (2, 1), stride= (1, 1), groups= 4, bias= False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size= (1, 4), stride= (1, 4), padding= 0),
            nn.Dropout(0.45)
        )

        self.separable_conv2d = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size= (1, 15), stride= (1, 1), padding= (0, 7), bias= False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size= (1, 8), stride= (1, 8), padding= 0),
            nn.Dropout(0.45),
        )

        

        self.fc = nn.Sequential(
            nn.Linear(184, 2)
        )

    def forward(self, x):
        x = self.first_conv2d(x)
        x = self.depthwise_conv2d(x)
        x = self.separable_conv2d(x)
        x = x.flatten(1)
        return self.fc(x)       