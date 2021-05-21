import tqdm
import torch


def train_one_epoch(net, dataloader, opt, batch_size=512, lr_sch=None, epoch=None):
    """
    This function is used to train a net with a specific dataset for one epoch.
    :param net:           torch.nn.Module
    :param dataloader:      torchvision.data.utils.Dataloader
    :param opt:             optimizer
    :param batch_size:      batch size
    :param lr_sch:          weather to use learning rate scheduler
    :param epoch:           how many epochs to train
    :return:                top1_rate & top5_rate, in tuple format
    """
    # Set net to train mode
    net.train()
    bar = tqdm.tqdm(dataloader)
    loss_fn = torch.nn.CrossEntropyLoss()

    top1_correct = 0
    top5_correct = 0
    top1_rate = 0
    top5_rate = 0
    for batch_ind, (batch_data, batch_label) in enumerate(bar):
        # Put data input gpu
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()

        # reset net gradient
        net.zero_grad()
        # Calculate loss
        predict = net(batch_data)
        loss = loss_fn(predict, batch_label)

        # back propagate loss and optimize
        loss.backward(retain_graph=True)
        opt.step()

        # Calculate top1 & top5 accuracy
        top1 = torch.argmax(predict, dim=1)
        top1_correct += torch.eq(top1, batch_label).sum().item()
        top1_rate = top1_correct / (batch_size * batch_ind + batch_label.numel())

        _, top5 = torch.topk(predict, k=5, dim=1)
        top5_correct += torch.eq(top5, batch_label.view(-1, 1)).sum().item()
        top5_rate = top5_correct / (batch_size * batch_ind + batch_label.numel())

        cur_lr = opt.state_dict()['param_groups'][0]['lr']

        bar.set_postfix_str(
            f"TRAIN {epoch} {top1_rate * 100:2.2f}% {top5_rate * 100:2.2f}% {cur_lr:.3e}")
    if lr_sch:
        lr_sch.step()
    return top1_rate, top5_rate
