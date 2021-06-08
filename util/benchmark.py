import torch

from util.cifar import CIFAR10, CIFAR100OE


def cifar10(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    cifar10 = CIFAR10(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    cifar100 = CIFAR100OE(root=config.data_path)

    train_loader = torch.utils.data.DataLoader(dataset=cifar10.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=cifar10.test_set,
        batch_size=100,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)

    return train_loader, oe_loader, val_loader
