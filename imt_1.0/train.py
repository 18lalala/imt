"""
Author: Ljy
Date: Aug 2022
"""

import argparse
import os
from tracemalloc import start
from dataset import collate_fn, imtdataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
from utils import get_loss
from imt import IMT
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

writer = SummaryWriter('./tblog')

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='imt', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--len_feature', type=int, default=32, help='length of features [default: 32]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=0, help='area for test [default: 0]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('imt'+'batch_size'+str(args.batch_size)+'__len_fea'+str(args.len_feature))
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/media/puzek/sda/dataset/imt_data/dataset/sequences/'
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = imtdataset(split='train', data_root=root, test_area=args.test_area)
    print("start loading test data ...")
    TEST_DATASET = imtdataset(split='test', data_root=root, test_area=args.test_area)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  collate_fn=collate_fn)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True,collate_fn=collate_fn)


    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = IMT()
    model = MODEL.cuda()
    criterion = get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_loss = 100

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        loss_ax_sum = 0
        loss_the_sum = 0
        loss_tr_sum = 0

        model = model.train()
        start = time.time()

        for i, (source, target, pose) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            source = source.data.numpy()
            target = target.data.numpy()
            pose_gt = pose.data.numpy()
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            pose_gt = torch.Tensor(pose_gt)
            source, target, pose_gt= source.float().cuda(), target.float().cuda(), pose.float().cuda()

            pose_pred = model(source, target)

            loss, loss_ax, loss_the, loss_tr = criterion(pose_gt, pose_pred)

            loss.backward()
            optimizer.step()
            loss_sum += loss
            loss_ax_sum += loss_ax
            loss_the_sum += loss_the
            loss_tr_sum += loss_tr
        print(loss_sum/num_batches)
        writer.add_scalar('loss_train', float(loss_sum/num_batches), epoch)
        writer.add_scalar('loss_ax_train', float(loss_ax_sum/num_batches), epoch)
        writer.add_scalar('loss_the_train', float(loss_the_sum/num_batches), epoch)
        writer.add_scalar('loss_tr_train', float(loss_tr_sum/num_batches), epoch)

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('time per item: %f s' %((time.time()-start)/len(TRAIN_DATASET)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            loss_ax_sum = 0
            loss_the_sum = 0
            loss_tr_sum = 0
            model = model.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (source, target, pose) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                source = source.data.numpy()
                target = target.data.numpy()
                pose = pose.data.numpy()
                source = torch.Tensor(source)
                target = torch.Tensor(target)
                pose = torch.Tensor(pose)
                source, target, pose= source.float().cuda(), target.float().cuda(), pose.float().cuda()
                pose_pred = model(source, target)
                loss, loss_ax, loss_the, loss_tr = criterion(pose_gt, pose_pred)

                loss_sum += loss
                loss_ax_sum += loss_ax
                loss_the_sum += loss_the
                loss_tr_sum += loss_tr

            writer.add_scalar('loss_test', float(loss_sum/num_batches), epoch)
            writer.add_scalar('loss_ax_test', float(loss_ax_sum/num_batches), epoch)
            writer.add_scalar('loss_the_test', float(loss_the_sum/num_batches), epoch)
            writer.add_scalar('loss_tr_test', float(loss_tr_sum/num_batches), epoch)

            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            mloss = loss_sum / float(num_batches)
            if  mloss <= best_loss:
                best_loss = mloss
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'mloss': mloss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mloss: %f' % mloss)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)