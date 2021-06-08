import torch
import torch.nn as nn
import torchvision

import functools; print = functools.partial(print, flush=True)
import numpy as np
import os
import time

from nn.helpers.metrics import accuracy
from nn.resnet import resnet26
from sklearn.metrics import average_precision_score, roc_auc_score
from util.helpers.log import Log
from util.helpers.setup import checkpoint, make_dirs, newline, save_model_info, to_gpu
from util.benchmark import cifar10
from util.parser import get_default_parser

to_list = lambda t: t.cpu().data.numpy().tolist()


def main():
    torch.backends.cudnn.benchmark = True

    parser = get_default_parser()
    config = parser.parse_args()

    make_dirs(config.ckpt_path)
    out = open(os.path.join(config.ckpt_path, "console.out"), "w")

    if config.dataset == "cifar10":
        train_loader, oe_loader, val_loader = cifar10(config)
    else:
        raise NotImplementedError

    save_model_info(config, file=out)

    f = resnet26(config, 1)
    f.cuda()

    if config.model == "adib":
        theta_0 = f.params()
    
    loss = nn.BCEWithLogitsLoss()    
    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, f.parameters()),
        lr=config.lr_sgd,
        momentum=config.momentum_sgd,
        weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim,
        milestones=list(map(int, config.milestones.split(","))),
        gamma=config.gamma)

    log = Log(file=out)
    log.register("time", format="{0:.4f}")
    log.register("loss", format="{0:.3f}")
    log.register("ap", format="{0:.3f}", color="yellow")
    log.register("auc", format="{0:.3f}", color="red")
    log.legend()

    for epoch in range(config.num_epochs):
        for i, batch in enumerate(zip(train_loader, oe_loader)):

            f.train()
            f.zero_grad()

            t = time.time()

            x = torch.cat((batch[0][0], batch[1][0]), 0)
            semi_targets = torch.cat((batch[0][1], batch[1][1]), 0)

            x, semi_targets = to_gpu(x, semi_targets)

            logits = f(x).squeeze()
            l = loss(logits, semi_targets.float())

            if config.model == "adib":
                l += config.alpha * torch.norm(f.params(backprop=True) - theta_0, 2)

            l.backward()
            optim.step()

            log.update("time", time.time() - t)
            log.update("loss", l.item(), x.size(0))
            log.report(which=["time", "loss"], epoch=epoch, batch_id=i)

        sched.step()
        newline(f=out)

        labels_scores = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):

                f.eval()

                x, labels = batch

                x, labels = to_gpu(x, labels)

                scores = torch.sigmoid(f(x)).squeeze()

                labels_scores += list(zip(to_list(labels), to_list(scores)))

        labels, scores = zip(*labels_scores)
        labels, scores = np.array(labels), np.array(scores)

        ap = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)

        log.update("ap", ap)
        log.update("auc", auc)
        log.report(which=["ap", "auc"], epoch=epoch, batch_id=i, newline=True)
        log.save_to_dat(epoch, config.ckpt_path)
        log.reset()

        if config.ckpt:
            checkpoint(config.ckpt_path, f)


if __name__ == "__main__":
    main()
