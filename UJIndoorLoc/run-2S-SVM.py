from argparse import ArgumentParser
import torch
from Dataset import ILDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import random
from tensorboardX import SummaryWriter
import pandas as pd
import os
from model_2S import AutoEncoder
from sklearn import svm


def loss_coding(y, y_hat):
    res = 0
    for i in range(len(y)):
        res += torch.mean((y - y_hat) ** 2)
    return res/y.shape[0]


def data_collecate(Dataset, Dataloader):
    Fs, Ls = np.zeros([len(Dataset), args.coding_dim]), np.zeros([len(Dataset)])
    for i, (feature, label) in enumerate(Dataloader):
        coding_vec = AutoEncoder(feature.to(device))[0].detach().cpu()
        b = feature.shape[0]
        Fs[i * b:(i + 1) * b] = coding_vec
        Ls[i * b:(i + 1) * b] = label
    return Fs, Ls


if __name__ == '__main__':
    parser = ArgumentParser(description='Auto-Encoder for Indoor Location')
    parser.add_argument("--seed", type=int, default=20220509)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--lr_AE', default=0.00001, type=float)
    parser.add_argument('--hidden_nodes', default=286, type=int,
                        help='hidden_nodes for MLP in Encoder/Decoder')
    parser.add_argument('--coding_dim', default=64, type=int,
                        help='dims of coding vector')
    parser.add_argument('--pretrain_epochs', default=10, type=int,
                        help='pretrain for auto-encoder')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='weight for loss')
    parser.add_argument('--commit', type=str, default='2S-AE-SVM',
                        help='commit for logs')
    args = parser.parse_args(args=[])

    device = torch.device("cpu")

    if not os.path.exists('weights'):
        os.makedirs('weights')
    trained_model_file1 = 'weights/lr{}-hn{}-cd{}-ptep{}-AE-SVM.pth'.\
        format(args.lr_AE, args.hidden_nodes, args.coding_dim, args.pretrain_epochs)

    # plot init
    writer = SummaryWriter(log_dir='logs/2S-lr{}-hn{}-cd{}-ptep{}-bs{}-{}'
                           .format(args.lr_AE, args.hidden_nodes, args.coding_dim,
                                   args.pretrain_epochs, args.batch_size, args.commit))

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # data loading & split & dataset init
    # oriTrainDataset = ILDataset('trainingData.csv')
    # TrainDataset, ValDataset = torch.utils.data.random_split(
    #     oriTrainDataset, [int(0.8*len(oriTrainDataset)), len(oriTrainDataset)-int(0.8*len(oriTrainDataset))],
    #     generator=torch.Generator().manual_seed(args.seed))
    # TestDataset = ILDataset('validationData.csv')
    data1 = pd.read_csv('trainingData.csv').to_numpy()
    data2 = pd.read_csv('validationData.csv').to_numpy()
    data = np.concatenate([data1, data2])
    ILDataset = ILDataset(data)
    TrainDataset, ValDataset, TestDataset = torch.utils.data.random_split(
        ILDataset, [int(0.6*len(ILDataset)), int(0.2*len(ILDataset)),
                    len(ILDataset)-int(0.2*len(ILDataset))-int(0.6*len(ILDataset))],
        generator=torch.Generator().manual_seed(args.seed))

    # dataloader init
    trainLoader = DataLoader(TrainDataset, shuffle=True, batch_size=args.batch_size)
    valLoader = DataLoader(ValDataset, shuffle=True, batch_size=args.batch_size)
    testLoader = DataLoader(TestDataset, shuffle=True, batch_size=args.batch_size)

    # model & optimizer init
    AutoEncoder = AutoEncoder(hidden_nodes=args.hidden_nodes, coding_dim=args.coding_dim).to(device)
    optimizer1 = Adam(AutoEncoder.parameters(), lr=args.lr_AE, weight_decay=args.weight_decay)

    # CE init & initial flag setting
    loss_CE = nn.CrossEntropyLoss()
    best_acc = -1
    best_loss_AE = 999

    # auto-encoder learning---------------------------------------------------------------------------
    for epoch in range(args.pretrain_epochs):
        print('\n')
        print('-Epoch {}/{} for Auto-Encoder Learning...'.format(epoch + 1, args.pretrain_epochs))
        # train
        AutoEncoder.train()
        L = 0
        for i, (feature, label) in enumerate(trainLoader):
            feature = feature.to(device)
            label = label.to(device)
            out = AutoEncoder(feature)
            loss1 = loss_coding(feature, out[1])
            loss = loss1
            loss.backward()
            optimizer1.step()
            L = L + loss.item()
        train_coding_loss = L / (i + 1)
        print(' train coding loss:{:.4f}'.format(train_coding_loss))

        # val
        AutoEncoder.eval()
        acc_floor = 0
        acc_building = 0
        L = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(valLoader):
                feature = feature.to(device)
                label = label.to(device)
                out = AutoEncoder(feature)
                loss1 = loss_coding(feature, out[1])
                L = L + loss1.item()
            val_coding_loss = L / (i + 1)
            print(' val coding loss:{:.4f}'.format(val_coding_loss))

            if val_coding_loss < best_loss_AE:
                best_loss_AE = val_coding_loss
                torch.save(AutoEncoder.state_dict(), trained_model_file1)
                print(' ===> Saving best Auto-Encoder in epoch {}'.format(epoch+1))

        # plot
        writer.add_scalar("pretrain/train-loss_coding", train_coding_loss, epoch + 1)
        writer.add_scalar("pretrain/val-loss_coding", val_coding_loss, epoch + 1)

    # SVM for coding vec---------------------------------------------------------------------
    print('\nTraining SVM after freezing AE...')
    AutoEncoder.load_state_dict(torch.load(trained_model_file1))
    AutoEncoder.eval()

    # train & test data collection
    TrainFs, TrainLs = data_collecate(TrainDataset, trainLoader)
    TestFs, TestLs = data_collecate(TestDataset, testLoader)

    # SVM init
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # SVM training
    predictor.fit(TrainFs, TrainLs)
    # SVM test
    predict_lable = predictor.predict(TestFs)
    acc = np.mean(predict_lable == TestLs)

    print('Final best result of SVM: acc_loc_pred={:.4f}'.format(acc))

