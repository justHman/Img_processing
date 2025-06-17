from argparse import ArgumentParser

def get_args():
    parsers = ArgumentParser()
    parsers.add_argument('--root', '-r', type=int, default=1, help='Source')
    
    args = parsers.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args.root)
    print(type(args.root))