from argparse import ArgumentParser
import torch
from Dataset import ILDataset
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import os
from model_E2E import AutoEncoder
from model_E2E_deep import deepAutoEncoder


def loss_coding(y, y_hat):
    res = 0
    for i in range(len(y)):
        res += torch.mean((y - y_hat) ** 2)
    return res/y.shape[0]


if __name__ == '__main__':
    parser = ArgumentParser(description='End-to-End Auto-Encoder for Indoor Location')
    parser.add_argument("--seed", type=int, default=20220509)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--hidden_nodes', default=288, type=int,
                        help='hidden_nodes for MLP in Encoder/Decoder')
    parser.add_argument('--coding_dim', default=64, type=int,
                        help='dims of coding vector')
    parser.add_argument('--epochs', default=50, type=int,
                        help='train epochs for prediction head')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='weight for loss')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--head_mode', type=str, default='fc',
                        help='mode of Predict Head ("mlp" or "fc")')
    parser.add_argument('--commit', type=str, default='AE-tanh',
                        help='commit for logs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weight saving init
    if not os.path.exists('weights'):
        os.makedirs('weights')
    trained_model_file = 'weights/lr{}-hn{}-cd{}-ep{}.pth'.\
        format(args.lr, args.hidden_nodes, args.coding_dim, args.epochs)

    # plot init
    writer = SummaryWriter(log_dir='logs/E2E-lr{}-hn{}-cd{}-ep{}-bs{}-wd{}-{}-{}'
                           .format(args.lr, args.hidden_nodes, args.coding_dim,
                                   args.epochs, args.batch_size, args.weight_decay, args.head_mode, args.commit))

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # data loading & split & dataset init
    data1 = pd.read_csv('trainingData.csv').to_numpy()
    data2 = pd.read_csv('validationData.csv').to_numpy()
    data = np.concatenate([data1, data2])
    ILDataset = ILDataset(data)
    TrainDataset, ValDataset, TestDataset = torch.utils.data.random_split(
        ILDataset, [int(0.6 * len(ILDataset)), int(0.2 * len(ILDataset)),
                    len(ILDataset) - int(0.2 * len(ILDataset)) - int(0.6 * len(ILDataset))],
        generator=torch.Generator().manual_seed(args.seed))

    # dataloader init
    trainLoader = DataLoader(TrainDataset, shuffle=True, batch_size=args.batch_size)
    valLoader = DataLoader(ValDataset, shuffle=True, batch_size=args.batch_size)
    testLoader = DataLoader(TestDataset, shuffle=True, batch_size=args.batch_size)

    # model & optimizer init
    AutoEncoder = AutoEncoder(hidden_nodes=args.hidden_nodes, coding_dim=args.coding_dim, head_mode=args.head_mode).to(device)
    optimizer = Adam(AutoEncoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CE init & initial flag setting
    loss_CE = nn.CrossEntropyLoss()
    best_acc = -1
    # best_loss_AE = 999

    # model learning
    for epoch in range(args.epochs):
        print('\n')
        print('-Epoch {}/{} Learning...'.format(epoch + 1, args.epochs))
        # train
        AutoEncoder.train()
        L, L1, L2 = 0, 0, 0
        for i, (feature, label) in enumerate(trainLoader):
            feature = feature.to(device)
            label = label.to(device)
            out = AutoEncoder(feature)
            loss1 = loss_coding(feature, out[1])
            loss2 = loss_CE(out[0], label.long())
            loss = (1 - args.beta) * loss1 + args.beta * loss2
            loss.backward()
            optimizer.step()
            L = L + loss.item()
            L1 = L1 + loss1.item()
            L2 = L2 + loss2.item()
        train_loss = L / (i + 1)
        train_coding_loss = L1 / (i + 1)
        train_pred_loss = L2 / (i + 1)
        print(' train loss:{:.4f}'.format(train_loss))

        # val & test
        AutoEncoder.eval()
        acc_loc = 0
        with torch.no_grad():
            # val
            for i, (feature, label) in enumerate(valLoader):
                feature = feature.to(device)
                label = label.to(device)
                out = AutoEncoder(feature)
                pred_loc = torch.max(out[0], dim=1)[1]
                acc_loc += (pred_loc == label).sum().item()
            val_acc_loc = acc_loc / len(ValDataset)
            print(' val: acc_loc_pred={:.4f}'.format(val_acc_loc))

            # test
            acc_loc = 0
            for i, (feature, label) in enumerate(testLoader):
                feature = feature.to(device)
                label = label.to(device)
                out = AutoEncoder(feature)
                pred = out[0]
                pred_loc = torch.max(out[0], dim=1)[1]
                acc_loc += (pred_loc == label).sum().item()
            test_acc_loc = acc_loc / len(TestDataset)
            print(' test: acc_loc_pred={:.4f}'.format(test_acc_loc))

            # model save
            if val_acc_loc > best_acc:
                torch.save(AutoEncoder.state_dict(), trained_model_file)
                best_acc = val_acc_loc
                best_epoch = epoch
                best_test_acc_loc = test_acc_loc
                print(' ====> Saving best model in epoch {}'.format(epoch+1))

        # plot
        writer.add_scalar("train/loss", train_loss, epoch+1)
        writer.add_scalar("train/loss_coding", train_coding_loss, epoch + 1)
        writer.add_scalar("train/loss_pred", train_pred_loss, epoch + 1)
        writer.add_scalar("acc_loc/val", val_acc_loc, epoch+1)
        writer.add_scalar("acc_loc/test", test_acc_loc, epoch + 1)

    print('\n')
    print('Final best result in epoch {}: acc_loc_pred={:.4f}'
          .format(best_epoch+1, best_test_acc_loc))

