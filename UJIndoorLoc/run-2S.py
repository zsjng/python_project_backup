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
from model_2S import AutoEncoder, Predict_head, Predict_head2


def loss_coding(y, y_hat):
    res = 0
    for i in range(len(y)):
        res += torch.mean((y - y_hat) ** 2)
    return res/y.shape[0]


if __name__ == '__main__':
    parser = ArgumentParser(description='Auto-Encoder for Indoor Location')
    parser.add_argument("--seed", type=int, default=20220509)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--lr_AE', default=0.00001, type=float)
    parser.add_argument('--lr_PH', default=0.0001, type=float)
    parser.add_argument('--hidden_nodes', default=286, type=int,
                        help='hidden_nodes for MLP in Encoder/Decoder')
    parser.add_argument('--coding_dim', default=64, type=int,
                        help='dims of coding vector')
    parser.add_argument('--epochs', default=5, type=int,
                        help='train epochs for prediction head')
    parser.add_argument('--pretrain_epochs', default=5, type=int,
                        help='pretrain for auto-encoder')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='weight for loss')
    parser.add_argument('--commit', type=str, default='low-same_lr',
                        help='commit for logs')
    args = parser.parse_args()

    device = torch.device( "cpu")

    if not os.path.exists('weights'):
        os.makedirs('weights')
    trained_model_file1 = 'weights/lr{}-{}-hn{}-cd{}-ep{}-AE.pth'.\
        format(args.lr_AE, args.lr_PH, args.hidden_nodes, args.coding_dim, args.epochs)
    trained_model_file2 = 'weights/lr{}-{}-hn{}-cd{}-ep{}-PH.pth'. \
        format(args.lr_AE, args.lr_PH, args.hidden_nodes, args.coding_dim, args.epochs)

    # plot init
    writer = SummaryWriter(log_dir='logs/2S-lr{}-{}-hn{}-cd{}-ep{}-bs{}-{}'
                           .format(args.lr_AE, args.lr_PH, args.hidden_nodes, args.coding_dim,
                                   args.epochs, args.batch_size, args.commit))

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
    Predict_head = Predict_head(coding_dim=args.coding_dim).to(device)
    optimizer1 = Adam(AutoEncoder.parameters(), lr=args.lr_AE, weight_decay=args.weight_decay)
    optimizer2 = Adam(Predict_head.parameters(), lr=args.lr_PH, weight_decay=args.weight_decay)

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
        Predict_head.eval()
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

    # predict head learning---------------------------------------------------------------------
    AutoEncoder.load_state_dict(torch.load(trained_model_file1))
    print('\nTraining predict head after frezing AE...')
    AutoEncoder.eval()
    for epoch in range(args.epochs):
        print('\n')
        print('-Epoch {}/{} Predict Head Learning...'.format(epoch + 1, args.epochs))
        # train
        Predict_head.train()
        L = 0
        for i, (feature, label) in enumerate(trainLoader):
            feature = feature.to(device)
            label = label.to(device)
            out = AutoEncoder(feature)
            pred = Predict_head(out[0])
            loss = loss_CE(pred, label.long())
            loss.backward()
            optimizer2.step()
            L = L + loss.item()
        train_loss = L / (i + 1)
        print(' train loss:{:.4f}'.format(train_loss))

        # val
        Predict_head.eval()
        acc_loc = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(valLoader):
                feature = feature.to(device)
                label = label.to(device)
                out = AutoEncoder(feature)
                pred = Predict_head(out[0])
                pred_loc = torch.max(out[0], dim=1)[1]
                acc_loc += (pred_loc == label).sum().item()
            val_acc_loc = acc_loc / len(ValDataset)
            print(' val: acc_loc_pred={:.4f}'.format(val_acc_loc))

            # test
            acc_floor = 0
            acc_building = 0
            for i, (feature, label) in enumerate(testLoader):
                feature = feature.to(device)
                label = label.to(device)
                out = AutoEncoder(feature)
                pred = Predict_head(out[0])
                pred_loc = torch.max(pred, dim=1)[1]
                acc_loc += (pred_loc == label).sum().item()
            test_acc_loc = acc_loc / len(TestDataset)
            print(' test: acc_loc_pred={:.4f}'.format(test_acc_loc))

            if val_acc_loc > best_acc:
                torch.save(Predict_head.state_dict(), trained_model_file2)
                best_acc = val_acc_loc
                best_epoch = epoch
                best_test_acc_loc = test_acc_loc
                print(' ====> Saving best model in epoch {}'.format(epoch+1))

        # plot
        writer.add_scalar("train/loss", train_loss, epoch + 1)
        writer.add_scalar("acc_loc/val", val_acc_loc, epoch + 1)
        writer.add_scalar("acc_loc/test", test_acc_loc, epoch + 1)

    print('\n')
    print('Final best result in epoch {}: acc_loc_pred={:.4f}'
          .format(best_epoch + 1, best_test_acc_loc))

