from argparse import ArgumentParser
from model import Simple_CNN
import torch
import cv2 
from torchvision.transforms import ToTensor
from torch import cuda, no_grad, nn, argmax

def get_args():
    parsers = ArgumentParser()
    parsers.add_argument('--size', '-s', type=int, default=32)
    parsers.add_argument('--check_point', '-cp', type=str, default='trained_models\\accuraciest.pt')
    parsers.add_argument('--path', '-p', type=str, default=None)
    parsers.add_argument('--num_classes', '-n', type=int, default=10)

    args = parsers.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        print('Cuda is available!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = Simple_CNN(num_classes=args.num_classes).to(device)
    try:
        print('Loading model!')
        check_point = torch.load(args.check_point)
        model.load_state_dict(check_point['model'])
    except FileNotFoundError as e:
        raise e

    org_img = cv2.imread(args.path)
    if org_img is None:
        raise FileNotFoundError(f"Cannot load image at path: {args.path}")
    
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.size, args.size))
    img = ToTensor()(img)
    img = img[None, :, :, :]
    img = img.to(device)

    model.eval()
    with no_grad():
        outputs = model(img)
        probs = nn.Softmax()(outputs)
        max_pred = argmax(outputs, dim=1)

    animals = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
    cv2.imshow(f'This is {animals[max_pred]}: {probs[0, max_pred].item() * 100:.3f}%', org_img)
    cv2.waitKey(0)



    