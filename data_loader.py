from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from datasets import CIFAR_Dataset, Animal_Dataset
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose

if __name__ == '__main__':
    # dataset = CIFA_Dataset('data\\cifar-10-batches-py', train=True, transform=ToTensor())
    # dataset = CIFAR10(root='data', train=True, transform=ToTensor())
    transform = Compose([
        ToTensor(),
        Resize((200, 200))
    ])
    dataset = Animal_Dataset('data\\animal', train=True, transform=transform)
    # data, label = dataset.__getitem__(1)
    # print(data)
    # print(type(data))
    # print(data.shape)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=9,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    arr2 = []
    for data, label in data_loader:
        print(data)
        print(label)
        arr2.extend(label.data)
        break
    print(arr2)