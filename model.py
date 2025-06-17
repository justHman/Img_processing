from torch import nn
import torch

class Simple_NN(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class Simple_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.make_block_conv(3, 8)
        self.conv2 = self.make_block_conv(8, 8)
        self.flatten = nn.Flatten()
        self.fc1 = self.make_block_fc(8*8*8, 32)
        self.drout = nn.Dropout(p=0.5)
        self.fc2 = self.make_block_fc(32, num_classes)

    def make_block_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def make_block_fc(self, in_feature, out_feature):
        return nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=out_feature),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    number_class = 10
    model = Simple_NN(num_classes=number_class)
    input = torch.rand(8, 3, 32, 32)
    # output = model(input)
    # print(output.shape)
    # print(torch.argmax(output, dim=1))
    sd = model.state_dict()
    print(sd)
    print(sd.keys())
    print(sd.items())
    for key, value in sd.items():
        print(key, value.shape)
   
    
    

