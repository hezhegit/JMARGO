import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from evaluation import fmax, compute_roc
from tqdm import tqdm

from lr_adjust import CustomLRAdjuster
from utils import now

warnings.filterwarnings("ignore")

gamma = -100




def train(args, net, criterion, train_loader, valid_loader, test_loader, ckpt_dir, F_txt, device, kf):

    lr_sched = True
    # Define optimizer weight_decay=0.00005
    optimizer = optim.Adam(net.parameters(), lr=0.0007)
    # Define scheduler for learning rate adjustment
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = CustomLRAdjuster(optimizer, threshold1=0.003, threshold2=0.002, decline1=0.0001, decline2=0.0002)
    # Load checkpoint model and optimizer
    best_epoch = 0
    # best_epoch = load_checkpoint(net, optimizer, scheduler, filename=ckpt_dir + '/model_test.pth.tar')
    if best_epoch != 0:
        lowest_valid_loss, valid_auc, valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader)
        lowest_test_loss, test_auc, best_test_fmax, _, _ = evaluate(device, net, criterion, test_loader)
        print(f"--- best_epoch : {best_epoch} \t Lowest Valid Loss : {lowest_valid_loss} ---")
    else:
        lowest_valid_loss = 1.0
        best_test_fmax = -1

    start_time = time.time()
    valid_loss_save = []


    for epoch in range(best_epoch, args.num_epochs):
        net.train()
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print('--- Current learning rate: ', param_group['lr'])

        print(
            f"[*] Epoch:{epoch + 1} \t Namespace:{args.namespace.upper()} \t net:{args.net_type} ")
        train_loss_all = 0.0
        pbar = tqdm(train_loader)
        for data in pbar:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y
            with torch.set_grad_enabled(True):
                # Set the parameter gradients to zero
                optimizer.zero_grad()
                # Forward pass
                outputs = net(data)

                current_loss = criterion(outputs, labels.float())
                # corr_loss = correlation_loss(outputs)
                # Backward pass and optimize
                # total_loss = current_loss + corr_loss
                current_loss.backward()
                optimizer.step()
                train_loss_all += current_loss.item() / len(train_loader)
            pbar.set_description("Train Loss: %g" % round(current_loss.item(), 4))

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

        valid_loss, valid_auc, valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader)
        test_loss, test_auc, test_fmax, _, _ = evaluate(device, net, criterion, test_loader)

        valid_loss_save.append(valid_loss)
        if valid_loss < lowest_valid_loss:
            best_epoch = epoch + 1
            lowest_valid_loss = valid_loss
            torch.save(state, ckpt_dir + f'/model_fold_{kf}.pth.tar')

        print(
            "Epoch:[{}]\t Train Loss:{:.4f}\t Valid loss:{:.4f}\t Valid ROC AUC:{:.4f}\t Valid F-score:{:.4f}\tTest loss:{:.4f}\tValid ROC AUC:{:.4f}\tTest "
            "F-score:{:.4f}".format(
                epoch + 1, train_loss_all, valid_loss, valid_auc, valid_fmax, test_loss, test_auc, test_fmax),
            file=F_txt)
        F_txt.flush()

        print("--- Best_epoch:    {:.2f}\t          Lowest Loss:{:.4f}".format(best_epoch, lowest_valid_loss))
        print("--- valid Loss:                  %.4f" % valid_loss)
        print("--- valid roc auc score:         %.4f" % valid_auc)
        print("--- valid max F-score:           %.4f" % valid_fmax)
        print("--- test max F-score:           %.4f" % test_fmax)
        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(valid_loss)

    os.makedirs(ckpt_dir, exist_ok=True)
    Loss = np.array(valid_loss_save)
    file_path = os.path.join(ckpt_dir, "{}_Epoch_{}.npy".format(kf, args.num_epochs))
    np.save(file_path, Loss)
    # plot_loss(args.num_epochs, save_dir=ckpt_dir)
    end_time = time.time()

    print("[*] Finish training.")
    print("*" * 20)
    print(
        f"{now()} finished; best_epoch: {best_epoch}; Lowest_Loss:{lowest_valid_loss},time/epoch:{(end_time - start_time) / args.num_epochs}")
    print(
        f"{now()} finished; best_epoch: {best_epoch}; Lowest_Loss:{lowest_valid_loss},time/epoch:{(end_time - start_time) / args.num_epochs}",
        file=F_txt)
    torch.cuda.empty_cache()
    return


def test(device, net, criterion, model_file, test_loader):
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)

    # Evaluate model
    test_loss, test_auc, test_fmax, y_true, y_pred_sigm = evaluate(device, net, criterion, test_loader, nth=10)

    # # Save predictions
    # if save_file is not None:
    #     pickle.dump({'y_true': y_true, 'y_pred': y_pred_sigm}, open(save_file, 'wb'))
    # Display evaluation metrics
    print("--- test loss:                  %.4f" % test_loss)
    print("--- test roc auc score:         %.4f" % test_auc)
    print("--- test max F-score:           %.4f" % test_fmax)

    return epoch_num, test_loss, test_auc, test_fmax, y_true, y_pred_sigm


def evaluate(device, net, criterion, eval_loader, nth=10):

    # Eval each sample

    net.eval()
    avg_loss = 0.0
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():  # set all 'requires_grad' to False
        pbar = tqdm(eval_loader)
        for data in pbar:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y
            # Forward pass
            outputs = net(data)
            current_loss = criterion(outputs, labels.float())
            # avg_loss += current_loss.item()
            avg_loss += current_loss.item() / len(eval_loader)

            y_true.append(labels.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(outputs).cpu().numpy().squeeze())
            pbar.set_description("Valid Loss: %g" % round(current_loss.item(), 4))

        # Calculate evaluation metrics
        # avg_loss = avg_loss / len(eval_loader)
        y_true = np.vstack(y_true)
        y_pred_sigm = np.vstack(y_pred_sigm)

        roc_auc = compute_roc(y_true, y_pred_sigm)
        # # Maximum F-score
        avg_fmax = fmax(y_true, y_pred_sigm, nrThresholds=nth)

    return avg_loss, roc_auc, avg_fmax, y_true, y_pred_sigm


def load_checkpoint(net, optimizer=None, scheduler=None, filename='model_best.pth.tar'):
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        #
        # torch.save(checkpoint,filename)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        start_epoch = 0
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch


def plot_loss(n, save_dir):
    y = []
    enc = np.load(save_dir + "Epoch_{}.npy".format(n))
    tempy = list(enc)
    y += tempy

    x = range(0, len(y))
    plt.plot(x, y, ",-")
    plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.00005'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('LOSS')
    plt.savefig(save_dir + "Epoch_{}".format(n))
