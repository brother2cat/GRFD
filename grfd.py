import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse
import logging
import csv
import sys
import h5py

sys.path.append("..")

from model.VAEC import VAEC_cifar
from model.VAEC import WideResNet
from utils.setting import *
from datasets.get_dataset import get_dataset
from utils.local_linearity import cal_simOllg
from utils.utils import Norm, Disnorm
from utils.utils import h5py_dataset_adv, h5py_dataset_zandRx, h5py_dataset_adv_distan


def main():
    args = parser.parse_args()

    dir_name = f"judge{args.w_judge}_classify{args.w_classify}_robust{args.w_robust}"
    if not os.path.isdir("exdata/" + dir_name):
        os.mkdir("exdata/" + dir_name)
    path_name = "exdata/" + dir_name

    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)
    # file Handler
    fileHandler = logging.FileHandler("exdata/" + dir_name + '/debug.log')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    # consoleHandler
    consoleHandler = logging.StreamHandler(sys.stderr)
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)
    # overall logger level should <= min(handler) otherwise no log will be recorded.

    # disable other debug, since too many debug
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    logger.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

    train_dataset, test_dataset, test_num, label_class = get_dataset("cifar10", args.batchsize)
    logger.debug(f"load dataset cifar10")

    logger.info(f"--------------Begin the training of GRFD--------------")

    norm = Norm("cifar10")
    disnorm = Disnorm("cifar10")
    device = torch.device(args.device)
    model = VAEC_cifar(args.fm_channel, args.z_dim,
                       with_classifier=True, widen_factor=args.widen_factor, drop_rate=args.drop_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.lr_optim == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=args.lr_min, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 200, 260], gamma=0.1, last_epoch=-1)

    logger.info("Begin to train GRFD")
    for epoch in range(args.epochs):
        grfd_train(args, epoch, model, optimizer, train_dataset, device, norm, disnorm, logger, path_name)
        scheduler.step()

    torch.save(model.state_dict(), "exdata/" + dir_name + "/weight.pt")
    logger.debug(f"save the CVAE model as exdata/{dir_name}/weight.pt")
    classifier_acc("exdata/" + dir_name + "/weight.pt", test_dataset, test_num, logger)

    logger.info("--------Begin to disentangle the robust features of train dataset and test dataset--------")

    logger.debug(f"load the GRFD model from {path_name}/weight.pt")
    model.load_state_dict(torch.load(f"{path_name}/weight.pt"))
    distangle_clean_features(model, train_dataset, path_name + "/train.hdf5", device, logger)
    distangle_clean_features(model, test_dataset, path_name + "/test.hdf5", device, logger)
    for attack_method in ["fgsm", "pgd", "AutoAttack", "eotpgd", "vmifgsm"]:
        adv_path = "exdata/adversarial_examples/" + attack_method + ".hdf5"
        dataset = h5py_dataset_adv(adv_path)
        if not os.path.isdir(path_name + "/adv_distangle"):
            os.mkdir(path_name + "/adv_distangle")
        dataset = DataLoader(dataset, batch_size=256, shuffle=False)
        root = path_name + "/adv_distangle/" + attack_method + ".hdf5"
        distangle_adv_features(model, dataset, root, device, logger)

    logger.info("------Begin to train the target model with disentangled features---------")

    logger.debug('Load the classify disentangled train dataset and test dataset')
    classify_train_dataset = h5py_dataset_zandRx(path_name + "/train.hdf5")
    classify_train_dataset = DataLoader(classify_train_dataset, batch_size=args.classify_batchsize, shuffle=True)
    classify_test_dataset = h5py_dataset_zandRx(path_name + "/test.hdf5")
    classify_test_dataset = DataLoader(classify_test_dataset, batch_size=args.classify_batchsize, shuffle=False)

    classify_model = WideResNet("cifar10", 28, 10, args.widen_factor, args.drop_rate).to(device)
    classify_criterion = nn.CrossEntropyLoss().to(device)
    classify_optimizer = optim.AdamW(classify_model.parameters(), lr=args.classify_lr, betas=(0.9, 0.999),
                                     weight_decay=1.e-6)
    if args.classify_lr_optim == 'cosine':
        classify_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classify_optimizer, T_max=50, eta_min=args.classify_lr_min, last_epoch=-1)
    else:
        classify_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            classify_optimizer, milestones=[100, 200, 260], gamma=0.1, last_epoch=-1)
    logger.info('Begin to train the classify model')
    for epoch in range(args.classify_epochs):
        classify_train_and_test_one_epoch(args, epoch, classify_model, classify_optimizer, classify_criterion,
                                          classify_train_dataset, classify_test_dataset, device, logger, path_name)
        for attack_method in ["fgsm", "pgd", "AutoAttack", "eotpgd", "vmifgsm"]:
            adv_disentangled_path = path_name + "/adv_distangle/" + attack_method + ".hdf5"
            adv_disentangled_test_dataset = h5py_dataset_adv_distan(adv_disentangled_path)
            adv_disentangled_test_dataset = DataLoader(adv_disentangled_test_dataset,
                                                       batch_size=args.classify_batchsize, shuffle=False)
            acc_test_adv_disentangled(args, attack_method, classify_model, adv_disentangled_test_dataset,
                                      classify_criterion, device, logger)
        classify_scheduler.step()
    torch.save(classify_model.state_dict(), path_name + "/classfier.pt")
    logger.debug(f"Save the target as {path_name}/classfier.pt")


def grfd_train(args, epoch, model, optimizer, train_dataset, device, norm, disnorm, logger, path_name):
    model.train()
    total_loss_all = 0
    total_loss_recon = 0
    total_loss_kl = 0
    total_loss_classify = 0
    total_loss_robust = 0
    total_loss_judge = 0
    total_train_num = 0

    if args.change_w_recon:
        if epoch < int(args.epochs / 3):
            w_recon = 10 * args.w_recon
        elif epoch < int(args.epochs / 3 * 2):
            w_recon = 5 * args.w_recon
        else:
            w_recon = args.w_recon
    else:
        w_recon = args.w_recon
    w_kl = args.w_kl
    w_classify = args.w_classify
    w_robust = args.w_robust
    w_judge = args.w_judge

    start_time = time.time()

    for batch_idx, (x, label) in enumerate(train_dataset):
        x, label = x.to(device), label.to(device)
        classify_result, _, judge_result, recon_image, mu, logvar = model(x, False)

        loss_recon = F.mse_loss(recon_image, x)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kl /= args.batchsize * 3 * args.z_dim
        loss_classify = F.cross_entropy(classify_result, label)
        loss_judge = F.cross_entropy(judge_result, label)
        loss_robust = 1 - torch.mean(cal_simOllg(
            model, x, label, 0.01, norm, disnorm, device, backprop=True, cvae=True))
        loss_all = w_recon * loss_recon + w_kl * loss_kl + w_classify * loss_classify + w_robust * loss_robust + w_judge * loss_judge

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        total_loss_recon += loss_recon.cpu().detach().item() * x.shape[0]
        total_loss_kl += loss_kl.cpu().detach().item() * x.shape[0]
        total_loss_classify += loss_classify.cpu().detach().item() * x.shape[0]
        total_loss_robust += loss_robust.cpu().detach().item() * x.shape[0]
        total_loss_judge += loss_judge.cpu().detach().item() * x.shape[0]
        total_loss_all += loss_all.cpu().detach().item() * x.shape[0]
        total_train_num += x.shape[0]

    loss_recon_avg = total_loss_recon / total_train_num
    loss_kl_avg = total_loss_kl / total_train_num
    loss_classify_avg = total_loss_classify / total_train_num
    loss_robust_avg = total_loss_robust / total_train_num
    loss_judge_avg = total_loss_judge / total_train_num
    loss_all_avg = total_loss_all / total_train_num

    end_time = time.time()

    logger.info(f"epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']}, "
                f"loss_recon_avg: {loss_recon_avg}, loss_kl_avg:{loss_kl_avg}, "
                f"loss_classify_avg:{loss_classify_avg}, loss_robust_avg:{loss_robust_avg}, "
                f"loss_judge_avg:{loss_judge_avg}, loss_all_avg:{loss_all_avg}-----------------"
                f"train_time:{end_time - start_time}")
    with open(path_name + '/grfd_train_lr_loss.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, optimizer.param_groups[0]['lr'],
                         loss_recon_avg, loss_kl_avg, loss_classify_avg, loss_robust_avg, loss_judge_avg, loss_all_avg])


def classifier_acc(model_name, dataset, test_num, logger):
    args = parser.parse_args()
    device = torch.device(args.device)
    model = VAEC_cifar(args.fm_channel, args.z_dim,
                       with_classifier=True, widen_factor=args.widen_factor, drop_rate=args.drop_rate).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    acc_classify = 0
    acc_judge = 0
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(dataset):
            x, label = x.to(device), label.to(device)
            classify_result, _, judge_result, _, _, _ = model(x, False)
            pred = classify_result.argmax(dim=1)
            acc_classify += pred.eq(label).sum().float().item()
            pred = judge_result.argmax(dim=1)
            acc_judge += pred.eq(label).sum().float().item()
    logger.info(f"classify acc1:{acc_classify / test_num}")
    logger.info(f"judge acc1:{acc_judge / test_num}")


def distangle_clean_features(model, dataloader, root, device, logger):
    initial_image = torch.tensor([])
    target = torch.tensor([])
    latent_z = torch.tensor([])
    Rx_image = torch.tensor([])
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(dataloader):
            x = x.to(device)
            classify_result, z_sampling, _, recon_image, mu, logvar = model(x, False)
            Rx = x - recon_image
            initial_image = torch.cat([initial_image, x.cpu().detach()])
            target = torch.cat([target, label])
            latent_z = torch.cat([latent_z, z_sampling.cpu().detach()])
            Rx_image = torch.cat([Rx_image, Rx.cpu().detach()])

    train_file = h5py.File(root, "w")
    train_file.create_dataset("image", data=initial_image)
    train_file.create_dataset("target", data=target)
    train_file.create_dataset("latent_z", data=latent_z)
    train_file.create_dataset("Rx_image", data=Rx_image)
    train_file.close()
    logger.info(f'finish distangle and save as {root}')


def distangle_adv_features(model, dataloader, root, device, logger):
    initial_image = torch.tensor([])
    target = torch.tensor([])
    adv_image = torch.tensor([])
    latent_z = torch.tensor([])
    Rx_image = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, label, x_adv) in enumerate(dataloader):
            x_adv = x_adv.to(device)
            classify_result, z_sampling, _, recon_image, mu, logvar = model(x_adv, False)
            Rx_adv = x_adv - recon_image
            initial_image = torch.cat([initial_image, x.cpu().detach()])
            target = torch.cat([target, label])
            adv_image = torch.cat([adv_image, x_adv.cpu().detach()])
            latent_z = torch.cat([latent_z, z_sampling.cpu().detach()])
            Rx_image = torch.cat([Rx_image, Rx_adv.cpu().detach()])

    train_file = h5py.File(root, "w")
    train_file.create_dataset("image", data=initial_image)
    train_file.create_dataset("target", data=target)
    train_file.create_dataset("adv", data=adv_image)
    train_file.create_dataset("latent_z", data=latent_z)
    train_file.create_dataset("Rx_image", data=Rx_image)
    train_file.close()
    logger.info(f'finish distangle and save as {root}')


def classify_train_and_test_one_epoch(args, epoch, model, optimizer, criterion, train_dataset, test_dataset, device,
                                      logger, path_name):
    start_time = time.time()
    model.train()
    train_loss_all = 0
    train_data_num = 0
    for batch_idx, (_, label, _, Rx) in enumerate(train_dataset):
        Rx, label = Rx.to(device), label.to(device)
        logits = model(Rx)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_all += loss.cpu().detach().item() * label.shape[0]
        train_data_num += label.shape[0]
    train_loss_all_avg = train_loss_all / train_data_num

    model.eval()
    test_loss_all = 0
    test_data_num = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (_, label, _, Rx) in enumerate(test_dataset):
            Rx, label = Rx.to(device), label.to(device)
            logits = model(Rx)
            loss = criterion(logits, label)
            pred = logits.argmax(dim=1)

            test_loss_all += loss.cpu().detach().item() * label.shape[0]
            test_data_num += label.shape[0]
            acc += pred.eq(label).sum().float().item()

    test_loss_avg = test_loss_all / test_data_num
    acc = acc / test_data_num
    end_time = time.time()

    logger.info(f"epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']}, "
                f"train_loss_avg: {train_loss_all_avg}, test_loss_avg:{test_loss_avg}, "
                f"test_acc:{acc}, ------------------ time:{end_time - start_time}")
    with open(path_name + '/classify_train_lr_loss.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, optimizer.param_groups[0]['lr'], train_loss_all_avg, test_loss_avg, acc])


def acc_test_adv_disentangled(args, attack_method, model, test_dataset, criterion, device, logger):
    model.eval()
    test_loss_all = 0
    test_data_num = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (_, target, _, _, Rx) in enumerate(test_dataset):
            Rx, target = Rx.to(device), target.to(device)
            logits = model(Rx)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1)

            test_loss_all += loss.cpu().detach().item() * target.shape[0]
            test_data_num += target.shape[0]
            acc += pred.eq(target).sum().float().item()

    test_loss_avg = test_loss_all / test_data_num
    acc = acc / test_data_num
    logger.info(f"{attack_method}: test_loss: {test_loss_avg}, acc:{acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CD-VAE Training')
    parser.add_argument("-dv", '--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [mnist/cifar10/miniimagenet]')
    parser.add_argument('--fm_channel', default=32, type=int, help='CD-VAE channel num of  feature map after encode')
    parser.add_argument('--z_dim', default=2048, type=int, help='dim of latent variable z')
    parser.add_argument('--widen_factor', default=10, type=int, help='In CD-VAE, classifier widen factor')
    parser.add_argument('--drop_rate', default=0.3, type=float, help='In CD-VAE, classifier drop_rate')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=150, type=int, help='training_epochs')
    parser.add_argument('--lr_optim', default='cosine', type=str, help='learning rate change method')
    parser.add_argument('--lr', default=1.e-3, type=float, help='when lr_optim is cosine, the initial lr')
    parser.add_argument('--lr_min', default=1.e-4, type=float, help='when lr_optim is cosine, the min lr')
    parser.add_argument('--w_recon', default=1.0, type=float, help='reconstruction weight')
    parser.add_argument('--w_robust', default=1.0, type=float, help='robust weight')
    parser.add_argument('--w_kl', default=0.2, type=float, help='kl weight')
    parser.add_argument('--w_classify', default=0.2, type=float, help='with classifier, classify loss weight')
    parser.add_argument('--judge', default=True, type=bool, help='start judge or not')
    parser.add_argument('--w_judge', default=0.2, type=float, help='z judge loss weight')
    parser.add_argument('--change_w_recon', default=True,
                        help='change_w_recon for reconstruction term which helps for better convergence')

    parser.add_argument('--classify_batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--classify_epochs', default=150, type=int, help='training_epochs')
    parser.add_argument('--classify_lr_optim', default='cosine', type=str, help='learning rate change method')
    parser.add_argument('--classify_lr', default=1.e-3, type=float, help='when lr_optim is cosine, the initial lr')
    parser.add_argument('--classify_lr_min', default=1.e-4, type=float, help='when lr_optim is cosine, the min lr')

    main()
