from torch.utils.data import Dataset
import os
import pickle 
import numpy as np
import cv2 
from torchvision.transforms import ToTensor, Resize

class CIFAR_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.transform = transform 
        self.size = 0
        paths = None
        if train:
            paths = [os.path.join(root, f'data_batch_{i}') for i in range(1, 6)]
        else:
            paths = [os.path.join(root, 'test_batch')]
        self.datas = []
        self.labels = []
        for path in paths:
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                self.size += len(data[b'labels'])
                self.datas.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        label = self.labels[index]
        data = self.datas[index]
        data = np.reshape(data, (3, 32, 32)) 
        data = np.transpose(data, (1, 2, 0)) 
        if self.transform:
            data = self.transform(data)
        return data, label



class Animal_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.transform = transform 
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        self.categories = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
        self.datas, self.labels = [], []
        for i, category in enumerate(self.categories):
            path = os.path.join(root, category)
            for file in os.listdir(path):
                if file.endswith('.jpeg'):
                    self.datas.append(os.path.join(path, file))
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        path = self.datas[index]
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.transform:
            data = self.transform(data)
        return data, label
    
if __name__ == "__main__":
    # datas = CIFAR_Dataset('data\\cifar-10-batches-py', train=False, transform=ToTensor())
    # data, label = datas.__getitem__(11)
    # print(data)
    # print(type(data))
    # print(data.shape)
    # cv2.imshow('img', cv2.resize(data, (100, 100)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    datas = Animal_Dataset('data\\animal', train=True, transform=ToTensor())
    data, label = datas.__getitem__(111)
    print(data)
    print(type(data))
    print(data.shape)
    
    # cv2.imshow('img', cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()