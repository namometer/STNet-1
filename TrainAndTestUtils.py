import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

criterion_rec = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()


def train(model: nn.Module, trainLoader: DataLoader, epoch: int, auto: bool, optimizer: optim.Optimizer):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss_cls = 0.0
    label_0 = 0
    f1_score_avg = 0
    correct_0 = 0
    correct_1 = 0
    running_total = 0
    running_correct = 0
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels - 1
        labels[labels[:] <= 0] = 0
        labels = labels.to(device)

        optimizer.zero_grad()

        inputs = inputs.float()
        outputs = model(inputs, labels)
        loss_cls = criterion_cls(outputs, labels)
        loss = loss_cls.float()

        loss.backward()
        optimizer.step()

        running_loss_cls += loss_cls.item()
        _, predicted = torch.max(outputs.data, dim=-1)
        f1_score_item = f1_score(labels.cpu().numpy(), predicted.cpu().numpy()) * 100
        f1_score_avg += f1_score_item
        running_total += inputs.shape[0]
        running_correct += (predicted == labels).sum().item()
        for ind, label in enumerate(labels):
            if label == 0:
                label_0 += 1
                if predicted[ind] == 0:
                    correct_0 += 1
            if predicted[ind] == 1 and label == 1:
                correct_1 += 1

        if i == len(trainLoader) - 1:
            label_1 = running_total - label_0
            loss_cls = running_loss_cls / (i + 1)
            acc = 100 * running_correct / running_total
            acc_0 = 100 * correct_0 / label_0
            acc_1 = 100 * correct_1 / label_1
            f1_score_avg = f1_score_avg / (i + 1)

            print('[epoch %d]:\tcls=%.3f \tacc=%.2f \tacc_0=%.3f \tacc_1=%.3f \tf1_score=%.3f' % (
                    epoch + 1, loss_cls, acc, acc_0, acc_1, f1_score_avg))

            return acc, acc_0, acc_1, f1_score_item


def test(model: nn.Module, testLoader: DataLoader, num_class: int, optimizer: optim.Optimizer = None,
         save_path: str = None, epoch: int = 0, EPOCH: int = 0, goodcon = False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    f1_score_avg = 0
    class_correct = torch.zeros(num_class).to(device)
    class_total = torch.zeros(num_class).to(device)
    with torch.no_grad():  # 测试集不用算梯度
        for data in testLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels - 1
            labels[labels[:] <= 0] = 0
            labels = labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, dim=-1)
            comp = predicted.eq(labels)
            f1_score_item = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted', labels=np.unique(predicted.cpu().numpy())) * 100
            f1_score_avg += f1_score_item
            correct += comp.sum().item()
            for i in range(num_class):
                idx = labels.eq(i)
                class_total[i] += idx.sum()
                class_correct[i] += comp[idx].sum()
        acc = (correct / total) * 100
        class_acc = class_correct / class_total.clamp_min(1e-4) * 100
        f1_score_avg = (f1_score_avg / total) * 100
    print('[%d/%d]: Test performance: \tacc=%.2f \tf1_score=%.3f' % (epoch + 1, EPOCH, acc, f1_score_avg))
    print('\t\tClass accuracy: \t[' + ', '.join('%.2f' % class_acc[i].item() for i in range(num_class)) + ']')

    if epoch % 5 == 4 or abs(EPOCH - epoch) <= 10 or goodcon:
        writeChiDir = os.path.join(save_path, model.name)
        if not os.path.exists(writeChiDir):
            os.mkdir(writeChiDir)
        dirlist = os.listdir(writeChiDir)
        if epoch < 10:
            if len(dirlist) == 0:
                n = 1
            else:
                n = max([int(s.split('(')[0]) for s in dirlist]) + 1
        else:
            n = max([int(s.split('(')[0]) for s in dirlist])
        writeChiDir = os.path.join(writeChiDir, str(n) + model.para_str)
        if not os.path.exists(writeChiDir):
            os.mkdir(writeChiDir)

        torch.save(model.state_dict(),
                   os.path.join(writeChiDir, r'model(%d, %.1f).pth' % (epoch + 1, acc)))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(writeChiDir, r'optimizer(%d, %.1f).pth' % (
                epoch + 1, acc)))
    return acc
