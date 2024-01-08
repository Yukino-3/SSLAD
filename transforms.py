import torchvision
import torch

class Transforms:
    def __init__(self, size=32):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                #torchvision.transforms.RandomResizedCrop(size=96),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def mask_correlated_samples(args):
    mask = torch.ones((args.batch_size * 2, args.batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(args.batch_size):
        mask[i, args.batch_size + i] = 0
        mask[args.batch_size + i, i] = 0
    return mask