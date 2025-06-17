from datasets import CIFAR_Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import Simple_NN
from torch import nn, cuda, no_grad
from torch.optim import SGD
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

if __name__ == '__main__':
    train_datas = CIFAR_Dataset('data\\cifar-10-batches-py', train=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_datas,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_datas = CIFAR_Dataset('data\\cifar-10-batches-py', train=False, transform=ToTensor())
    test_loader = DataLoader(
        dataset=test_datas,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    model = Simple_NN(num_classes=10)
    if cuda.is_available():
        print('Cuda is available!')
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 10
    iters = len(train_loader)

    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_loader)
        for iter, (datas, labels) in enumerate(bar):
            if cuda.is_available():
                datas = datas.cuda()
                labels = labels.cuda()

            # Forward
            outputs = model(datas)
            loss = criterion(outputs, labels)

            # Backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            bar.set_description(f'Epoch: {epoch + 1}/{epochs} | Iter: {iter + 1}/{iters} | Loss: {loss:.3f}')

        model.eval()  # Chuyển model sang chế độ test
        all_labels = []
        all_predicts = []
        with no_grad():
            for iter, (datas, labels) in enumerate(test_loader):
                if cuda.is_available():
                    datas = datas.cuda()
                    labels = labels.cuda()

                # Forward
                outputs = model(datas)
                all_labels.extend([val.item() for val in labels])
                all_predicts.extend([val.item() for val in torch.argmax(outputs, dim=1)])

        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss}')
        rp = classification_report(all_labels, all_predicts)
        print(rp)
