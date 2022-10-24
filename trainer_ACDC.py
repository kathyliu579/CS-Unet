#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
# import matplotlib.pyplot as plt
from utils.utils import DiceLoss
from torch.utils.data import DataLoader
from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from tqdm import tqdm
import os
from torchvision import transforms
from utils.test_ACDC import inference
from medpy.metric import dc,hd95
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=15,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        warmup_epochs = int(epochs / 20)
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def trainer_acdc(args, model, snapshot_path):

    # if args.usecheckpoint:
    #     model.load_state_dict(torch.load(args.checkpoint))

    train_dataset = ACDCdataset(args.root_path, args.list_dir, split="train", transform=
                                       transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    db_val=ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    valloader=DataLoader(db_val, batch_size=1, shuffle=False)
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    save_interval = args.n_skip  # int(max_epoch/6)

    iter_num = 0
    Loss = []
    Test_Accuracy = []

    Best_dcs = 0.8
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='%(asctime)s   %(levelname)s   %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    max_iterations = args.max_epochs * len(Train_loader)

    base_lr = args.base_lr
    base_weight = args.base_weight
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_weight)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(Train_loader), args.max_epochs, warmup=True)
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0
        with tqdm(desc='Epoch %d - train' % (epoch),
                  unit='it', total=len(Train_loader)) as pbar:
            for i_batch, sampled_batch in enumerate(Train_loader):
                image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
                image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
                loss = loss_dice * 0.5+ loss_ce * 0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()
                lr_ = optimizer.param_groups[0]["lr"]
                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_

                iter_num = iter_num + 1


                train_loss += loss.item()
                # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
                pbar.set_postfix(loss=train_loss / (i_batch + 1), lr=lr_)
                pbar.update()
            # Loss.append(train_loss/len(train_dataset))

            # loss visualization

            # fig1, ax1 = plt.subplots(figsize=(11, 8))
            # ax1.plot(range(epoch + 1), Loss)
            # ax1.set_title("Average trainset loss vs epochs")
            # ax1.set_xlabel("Epoch")
            # ax1.set_ylabel("Current loss")
            # plt.savefig('loss_vs_epochs_gauss.png')

            # plt.clf()
            # plt.close()

        # ---------- Validation ----------
        if epoch > 50 and (epoch + 1) % save_interval == 0:
            dc_sum = 0
            model.eval()
            for i, val_sampled_batch in enumerate(valloader):
                val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
                val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
                    torch.FloatTensor)
                val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(
                    1), val_label_batch.cuda().unsqueeze(1)

                val_outputs = model(val_image_batch)
                val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
                dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
                avg_dcs = dc_sum / len(valloader)
            logging.info("Validation ===>avg_dsc: %f" % avg_dcs)

            if avg_dcs > Best_dcs:
                save_mode_path = os.path.join(snapshot_path, 'epoch={}_avg_dcs={}.pth'.format(epoch, avg_dcs))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                Best_dcs = avg_dcs
                # ---------- Test ----------
                avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir) #args.test_save_dir
                Test_Accuracy.append(avg_dcs)
            elif avg_dcs > 0.83:
                # ---------- Test ----------
                avg_dcs, avg_hd = inference(args, model, testloader, None)
                Test_Accuracy.append(avg_dcs)
                save_mode_path = os.path.join(snapshot_path, 'test_epoch={}_avg_dcs={}.pth'.format(epoch, avg_dcs))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            # val visualization
            # fig2, ax2 = plt.subplots(figsize=(11, 8))
            # ax2.plot(range(int((epoch + 1) // save_interval)), Test_Accuracy)
            # ax2.set_title("Average val dataset dice score vs epochs")
            # ax2.set_xlabel("Epoch")
            # ax2.set_ylabel("Current dice score")
            # plt.savefig('val_dsc_vs_epochs_gauss.png')
            # plt.clf()
            # plt.close()

        if epoch >= args.max_epochs - 1:
            save_mode_path = os.path.join(snapshot_path,  'epoch={}_lr={}.pth'.format(epoch, lr_))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # ---------- Test ----------
            avg_dcs, avg_hd = inference(args, model, testloader, None)
            Test_Accuracy.append(avg_dcs)
            print(max(Test_Accuracy))
            break
    return "Training Finished!"