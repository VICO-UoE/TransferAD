import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(description='transfer-ad')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--ckpt_path', type=str, default="ckpt")
    parser.add_argument('--ckpt', action="store_true")
    parser.add_argument('--data_path', type=str, default="data")

    parser.add_argument('--benchmark', type=str, default="one_vs_rest", choices=["hold_one_out", "one_vs_rest"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument('--model', type=str, default="adib", choices=["adib", "adra"])
    parser.add_argument('--normal_class', type=int, default=0, choices=range(10))
    parser.add_argument('--params_path', type=str, default="resnet26.pth")

    parser.add_argument('--alpha', type=float, default=0.01, help="regularization strength")
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr_sgd', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default="60,80")
    parser.add_argument('--momentum_sgd', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1.e-4)
    
    return parser
