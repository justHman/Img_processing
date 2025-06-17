from datasets import Animal_Dataset
from torch.utils.data import DataLoader
from model import Simple_CNN, Simple_NN
from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine
from torch.optim import SGD 
from torch import cuda, no_grad, nn, argmax
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
from argparse import ArgumentParser
import os
import shutil
import torch
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import cv2

def plot_confusion_matrix_to_tensorboard(y_true, y_pred, class_names, writer, global_step, tag="ConfusionMatrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm_normalized, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(tag)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf).convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

    writer.add_images(tag, image_tensor, global_step)


def get_args():
    parsers = ArgumentParser()
    parsers.add_argument('--loggin', '-l', type=str, default='tensorboard')
    parsers.add_argument('--batch_size', '-b', type=int, default=128)
    parsers.add_argument('--epochs', '-e', type=int, default=10)
    parsers.add_argument('--size', '-s', type=int, default=32)
    parsers.add_argument('--num_classes', '-n', type=int, default=10)
    parsers.add_argument('--trained_models', '-tr', type=str, default='trained_models')
    parsers.add_argument('--check_point', '-cp', type=str, default='trained_models\\lastest.pt')

    args = parsers.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if cuda.is_available():
        print('Cuda is available!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform_train = Compose([
        ToTensor(),
        Resize((args.size, args.size)),
        RandomAffine(
            degrees=15, 
            translate=(0.5, 0.5),
            scale=(0.8, 1.2),
            shear=10,
        )
    ])

    train_datas = Animal_Dataset(root='data\\animal', train=True, transform=transform_train)
    train_loader = DataLoader(
        dataset=train_datas,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Test augumentation
    img, _ = train_datas[2]
    print(f'Image shape: {img.shape}')
    img = img.permute(1, 2, 0).numpy() * 255.0
    img = img.astype(np.uint8)   
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    cv2.imshow('Sample Image', img)
    cv2.waitKey(0)
    exit(0)

    transform_test = Compose([
        ToTensor(),
        Resize((args.size, args.size))
    ])

    test_datas = Animal_Dataset(root='data\\animal', train=False, transform=transform_test)
    test_loader = DataLoader(
        dataset=test_datas,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    if os.path.exists(args.loggin):
        shutil.rmtree(args.loggin)  # Xóa cả folder và toàn bộ nội dung
    os.makedirs(args.loggin, exist_ok=True)
    os.makedirs(args.trained_models, exist_ok=True)

    writer = SummaryWriter(args.loggin)
    model = Simple_CNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=0.001, momentum=0.9)
    iters = len(train_loader)
    start = 0
    epochs = args.epochs 
    acraciest = 0

    if args.check_point:
        print(f'Loading {args.check_point.split("\\")[-1]} model!')
        check_point = torch.load(args.check_point)
        model.load_state_dict(check_point['model'])
        optim.load_state_dict(check_point['optim'])
        start = check_point['epoch']
        acraciest = check_point['acracy']

    if args.epochs < start:
        epochs = start + 1

    for epoch in range(start, epochs):
        model.train()
        train_preds, train_labels = [], []
        bar = tqdm(train_loader, colour='green')
        for i, (datas, labels) in enumerate(bar):
            datas = datas.to(device)
            labels = labels.to(device)
        
            outputs = model(datas)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            
            writer.add_scalar('Train/Loss/Iter', loss, epoch*iters + i)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            bar.set_description(f'Epoch: {epoch + 1}/{epochs} | Iter: {i + 1}/{iters} | Loss: {loss:.3f}')

        writer.add_scalar('Train/Loss/Epoch', loss, epoch)

        model.eval()
        val_labels = []
        val_preds = []
        with no_grad():
            for i, (datas, labels) in enumerate(test_loader):
                datas = datas.to(device)
                labels = labels.to(device)

                outputs = model(datas)
                val_labels.extend([val.item() for val in labels])
                val_preds.extend([val.item() for val in argmax(outputs, dim=1)])

        acracy = accuracy_score(val_labels, val_preds)
        print(f'Accuracy: {acracy:.3f}')
        writer.add_scalar('Val/Accuracy/Epoch', acracy, epoch)
        plot_confusion_matrix_to_tensorboard(
            train_labels, train_preds, test_datas.categories, writer, epoch, tag="Train/ConfusionMatrix"
        )

        plot_confusion_matrix_to_tensorboard(
            val_labels, val_preds, test_datas.categories, writer, epoch, tag="Val/ConfusionMatrix"
        )
        
        check_point = {
            'model': model.state_dict(),
            'epoch': epoch + 1,
            'optim': optim.state_dict(),
            'acracy': acracy
        }
        torch.save(check_point, f'{args.trained_models}/lastest.pt')
        if acracy > acraciest:
            acraciest = acracy
            print(f'Accuraciest: {acraciest}')
            torch.save(check_point, f'{args.trained_models}/accuraciest.pt')
            

