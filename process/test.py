import torch
import tqdm
import torch.utils.data

def test_once(model, dataloader, batch_size=512, epoch=None):
    model.eval()
    bar = tqdm.tqdm(dataloader)

    top1_correct = 0
    top5_correct = 0
    top1_rate = 0
    top5_rate = 0
    for batch_ind, (batch_data, batch_label) in enumerate(bar):
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        with torch.no_grad():
            predict = model(batch_data)

            top1 = torch.argmax(predict, dim=1)
            top1_correct += torch.eq(top1, batch_label).sum().item()
            top1_rate = top1_correct / (batch_size * batch_ind + batch_label.numel())

            _, top5 = torch.topk(predict, 5, 1)
            top5_correct += torch.eq(top5, batch_label.view(-1, 1)).sum().item()
            top5_rate = top5_correct / (batch_size * batch_ind + batch_label.numel())

            bar.set_postfix_str(f"TEST E {epoch} TOP1 {top1_rate * 100:2.2f}% TOP5 {top5_rate * 100:2.2f}%")
    return top1_rate, top5_rate

