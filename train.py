import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from module.Netstruc import STNet, TemNet
from utils.Dataset import Samples
from TrainAndTestUtils import train, test

if __name__ == '__main__':
    num_class = 2
    auto = False
    batch_size = 64 if not auto else 32
    EPOCH = 100
    res_path = r'output/train_network'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    model = STNet(d_model=1656, d_bq=64, d_q=64, d_v=64, n_heads=8,
                          d_ff=72, n_layers=5, n_class=num_class, dropout=0.1,
                          kernel=5, auto=auto)
    # model = TemNet(d_model=1656, d_bq=64, d_q=64, d_v=64, n_heads=8,
    #                  d_ff=72, n_layers=5, n_class=num_class, dropout=0.1,
    #                  kernel=5, auto=auto)
    model.to(device)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    train_path = r'data/scatter/train/'
    train_dataset = Samples(data_dir=[train_path])  # , without_labels=(10,), speed=(0.7, 1.3)

    test_path = r'data/scatter/test/'
    test_dataset = Samples(data_dir=[test_path])

    trainLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    x = np.arange(int(EPOCH/10))
    acc = np.zeros(int(EPOCH/10))
    acc_tr = np.zeros(EPOCH)
    acc_tr0 = np.zeros(EPOCH)
    acc_tr1 = np.zeros(EPOCH)
    f1_score = np.zeros(EPOCH)
    max_iter = 0
    print('model name: %s' % model.name)
    print('parameters: %s' % model.para_str)
    for epoch in range(EPOCH):
        acc_train, acc_train0, acc_train1, f1_score_epoch = train(model, trainLoader, epoch, auto, optimizer)
        acc_tr[epoch] = acc_train
        acc_tr0[epoch] = acc_train0
        acc_tr1[epoch] = acc_train1
        f1_score[epoch] = f1_score_epoch
        if acc_train > acc_tr[max_iter] and acc_train > 90 and epoch % 10 != 9:
            max_iter = epoch
            test(model, testLoader, num_class, optimizer, res_path, epoch, EPOCH, True)
        if epoch % 10 == 9:  #每训练10轮 测试1次
            acc_test = test(model, testLoader, num_class, optimizer, res_path, epoch, EPOCH)
            acc[int(epoch/10)] = acc_test
            if acc_test > acc_tr[max_iter] and acc_test > 90:
                max_iter = epoch

    writeChiDir = os.path.join(res_path, model.name)
    dirlist = os.listdir(writeChiDir)
    n = max([int(s.split('(')[0]) for s in dirlist])
    writeChiDir = os.path.join(writeChiDir, str(n) + model.para_str)
    np.save(os.path.join(writeChiDir, 'accTest.npy'), acc)
    np.save(os.path.join(writeChiDir, 'accTrain.npy'), acc_tr)

    plt.figure()
    plt.plot(x, acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.figure()
    plt.plot(range(EPOCH), f1_score)
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score On TestSet')
    plt.figure()
    plt.plot(range(EPOCH), acc_tr, range(EPOCH), acc_tr0, range(EPOCH), acc_tr1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TrainSet')
    plt.show()
