import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
import util
import os.path as osp
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import argparse
from STGCN_WZ import model
from time import process_time


parser = argparse.ArgumentParser()

parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--root', type = str, default = 'data/', help = 'tyson data root path')

parser.add_argument('--adj', type = str, default = 'Tyson/adj.csv', help = 'adj')
parser.add_argument('--adjdata', type = str, default = 'Tyson/adj_tyson_csv.csv', help = 'tyson data path')
parser.add_argument('--data', type = str, default = 'Tyson/new_speed_5 - order.csv', help = 'adj tyson data path')
parser.add_argument('--con_data', type = str, default = 'Tyson/new_construction_5 - order.csv', help = 'construction adj tyson data path')
parser.add_argument('--dis_data', type = str, default = 'Tyson/adj_1.csv', help = 'distance of each road')

parser.add_argument('--num_nodes', type = int, default = '131', help = 'number of nodes')
parser.add_argument('--T',type = int, default = '12', help = 'number of timesteps.')
parser.add_argument('--P',type = int, default = '6', help = 'number for predict.')

# Train
parser.add_argument('--feature', type = int, default = '3', help = 'number of input feature(construction, weather, etc.)')
parser.add_argument('--in_channels', type = int, default = '1', help = 'tyson data path')
parser.add_argument('--batch_size', type = int, default = '16', help = 'batch size')
parser.add_argument('--learning_rate', type = float, default = '0.001', help = 'learning rate')
parser.add_argument('--epochs', type = int, default = '50', help = 'number of epochs')
parser.add_argument('--save', type = str, default = 'save/', help = 'save path')
args = parser.parse_args(args=[])
args.device = None
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("The current CUDA:", torch.cuda.is_available(), args.device)

def train(train_loader):
    epoch_loss = []
    train_pred = []
    train_label = []
    for i, (x, y) in enumerate(train_loader, 0):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        normalization_label = y[:, :, :, 0].cpu().detach().numpy()

        # STGCN
        # out = net(A_wave, x)
        out = net(x)

        pred = util.re_normalization(out, mean[0], std[0])
        label = util.re_normalization(normalization_label, mean[0],std[0])

        loss = criterion(out, y[:, :, :, 0])
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().cpu().numpy())

        train_pred.append(pred.cpu().detach().numpy())
        train_label.append(label)

        if (i + 1) % 20 == 0:
            pred = np.concatenate(train_pred, axis = 0)
            label = np.concatenate(train_label, axis = 0)
            train_mae, train_rmse, train_mape, b = util.metric(pred,label)
            print("[epoch %d][%d/%d] loss: %.4f mae: %.4f rmse: %.4f " % (
            epoch, i + 1, len(train_loader), loss.item(), train_mae, train_rmse))

    train_mae, train_rmse, train_mape, b = util.metric(train_pred, train_label)

    return train_rmse, sum(epoch_loss)

if __name__ == '__main__':
    train_, val_, test_, test_time, A, mean, std, index= util.loadData(args)
    print(args)

    #train_loader
    train_loader = DataLoader(
        dataset = train_,
        batch_size = args.batch_size,

    )

    val_loader = DataLoader(
        dataset = val_,
        batch_size = args.batch_size,
    )

    test_loader = DataLoader(
        dataset = test_,
        batch_size = args.batch_size,
    )

    A_wave = util.A_wave(args, A)

    net = model(args, A_wave).to(args.device)
    # STGCN
    # net = Net(A_wave.shape[0],
    #             1,
    #             12,
    #             6).to(device=args.device)
    # net = model(args, A_wave).to(args.device)


    total_params = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of parameters is: %d" % total_params)
    print("Total number of trainable parameters is: %d" % train_params)
    path = osp.join(args.save, 'params')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay= 0)
    if osp.exists(path) is not True:
        initepoch = 0
        criterion = nn.MSELoss().cuda()
    else:
        print("Loading the checkpoint...")
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        initepoch = checkpoint['epoch']
        criterion = checkpoint['loss']

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

    train_rmse_set = []
    training_losses = []
    validation_losses = []
    validation_rmse_set = []
    validation_MAE_set = []
    best_val_loss = np.inf
    epoch_set = []
    running_time_set = []

    for epoch in range(initepoch, args.epochs):
        start = process_time()
        net.train()
        print('\n\nStarting epoch %d / %d' % (epoch + 1, args.epochs))
        print('Learning Rate for this epoch: {}'.format(str(exp_lr_scheduler.get_lr()[0])))
        exp_lr_scheduler.step()
        t_rmse, loss = train(train_loader)
        train_rmse_set.append(t_rmse)
        training_losses.append(loss)
        epoch_set.append(epoch)
        net.eval()


        # Run validation
        val_pred = []
        val_label = []
        val_losses = []
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader,0):
                x = x_val.cuda()
                y = y_val.cuda()

                normalization_label = y[:, :, :, 0].cpu().detach().numpy()

                #STGCN
                # out = net(A_wave, x)
                out = net(x)

                pred = util.re_normalization(out, mean[0], std[0])
                label = util.re_normalization(normalization_label, mean[0], std[0])

                loss = criterion(out, y[:, :, :, 0])
                val_losses.append(loss.detach().cpu().numpy())

                val_pred.append(pred.cpu().detach().numpy())
                val_label.append(label)
            val_loss = sum(val_losses)
            validation_losses.append(val_loss)
            val_pred_ = np.concatenate(val_pred, axis=0)
            val_label_ = np.concatenate(val_label, axis=0)

            val_mae, val_rmse, val_mape, b = util.metric(val_pred_, val_label_)


            validation_rmse_set.append(val_rmse)
            validation_MAE_set.append(val_mae)
            end = process_time()
            running_time_set.append(end - start)

            if val_rmse < best_val_loss:
                stop = 0
                best_val_loss = val_rmse
                params_filename = osp.join(args.save, 'params')
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': net.state_dict(),
                     'opt_state_dict': optimizer.state_dict(),
                     'loss': criterion
                     }, params_filename)
                print("[epoch %d] best val_rmse: %.4f\n" % (epoch, best_val_loss))
            else:
                stop = stop + 1
                print(stop)

            if stop >= 8:
                print('No improvement after 8 epochs, we stop early!')
                break

    # model analysis
    data_analysis = {'epoch': epoch_set,'running_time': running_time_set, 'training_loss':training_losses,
         'training_rmse': train_rmse_set, 'validation_loss':validation_losses, 'validation_rmse': validation_rmse_set, 'validation_mae': validation_MAE_set}
    data_analysis  = pd.DataFrame(data=data_analysis )
    data_analysis .to_csv('result/STGCN_6.csv', index = False)






    # test
    best_params_filename = osp.join(args.save, 'params')
    net.load_state_dict(torch.load(best_params_filename)['model_state_dict'])

    test_pred = []
    test_label = []
    test_losses = []
    with  torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_loader, 0):
            x = x_test.cuda()
            y = y_test.cuda()

            normalization_label = y[:, :, :, 0].cpu().detach().numpy()
            # out = net(A_wave, x)
            out = net(x)

            pred = util.re_normalization(out, mean[0], std[0])
            label = util.re_normalization(normalization_label, mean[0], std[0])

            loss = criterion(out, y[:, :, :, 0])
            test_losses.append(loss.detach().cpu().numpy())

            test_pred.append(pred.cpu().detach().numpy())
            test_label.append(label)

        test_pred_ = np.concatenate(test_pred, axis=0)
        test_label_ = np.concatenate(test_label, axis=0)

        test_mae, test_rmse, test_mape, b = util.metric(test_pred_, test_label_)

        print("mae: %.4f, rmse: %.4f, mape: %.4f \n" % (test_mae, test_rmse, test_mape))