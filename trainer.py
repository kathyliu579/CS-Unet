import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
from utils.utils import test_single_volume
from metrics import dice, cal_hausdorff_distance

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2 方差公式， var[]代表方差，E[]表示期望(平均值)
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for _, data in enumerate(loader):
        data = data['image']
        # print(data.shape)
        channels_sum += torch.mean(data, dim = [0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim = [0, 2, 3])
        num_batches += 1


    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std, num_batches

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


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # ---------- construct dataset ----------
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("The test iterations per epoch is: {}".format(len(testloader)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # print(get_mean_std(trainloader))
    # quit()
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # ---------- training ----------
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5E-3)
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(trainloader), args.max_epochs, warmup=True)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    #iterator = tqdm(, ncols=70)
    # save best model
    best_performance = 0.0

    for epoch_num in range(max_epoch):
        total_loss = 0
        model.train()
        with tqdm(desc='Epoch %d - train' % (epoch_num),
                  unit='it', total=len(trainloader)) as pbar:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()
                lr_ = optimizer.param_groups[0]["lr"]

                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_

                total_loss += loss.item()
                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                #logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

                pbar.set_postfix(loss=total_loss / (i_batch + 1), lr=lr_)
                pbar.update()

                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
        # print(lr_)
        # ---------- Validation ----------
        # if (epoch_num > 10) and (epoch_num + 1) % 5 == 0:
        # if (epoch_num > 158 and (epoch_num + 1) % 10 == 0) or (epoch_num >= max_epoch - 1):
        if  epoch_num > 128 and (epoch_num + 1) % 4 == 0:
        # if (epoch_num > 138 and (epoch_num + 1) % 10 == 0) or (epoch_num > 198 and (epoch_num + 1) % 5 == 0):
            model.eval()
            with torch.no_grad():
                metric_list = 0.0
                for j_batch, sample in enumerate(testloader):
                    h, w = sample["image"].size()[2:]
                    image, label, case_name = sample["image"], sample["label"], sample['case_name'][0]
                    metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                                  patch_size=[args.img_size, args.img_size],
                                                  test_save_path=None, case=case_name, z_spacing=args.z_spacing)
                    metric_list += np.array(metric_i)
                    # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                    # i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
                metric_list = metric_list / len(db_test)
                for i in range(1, args.num_classes):
                    logging.info(
                        'Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                logging.info(
                    'valid performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'best_epoch_' + str(epoch_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
        # ---------- save results ----------
        # save_interval = 10  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2 +20) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        if epoch_num >= max_epoch - 1:
            print(best_performance)
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     #iterator.close()
        #     break

    writer.close()
    return "Training Finished!"
