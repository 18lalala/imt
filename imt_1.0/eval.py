"""
Author: Ljy
Date: Aug 2022
"""

import argparse
from cgi import test
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


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='imt', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--len_feature', type=int, default=256, help='length of features [default: 32]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 16]')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--save_path', type=str, default=None, help='Save path [default: None]')
    parser.add_argument('--test_area', type=int, default=0, help='area for test [default: 0]')
    parser.add_argument('--pretrained_model', type=str, default=None, help='model path')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    log_string('PARAMETER ...')
    log_string(args)


    root = '/media/puzek/sda/dataset/imt_data/dataset/sequences/'
    BATCH_SIZE = args.batch_size

    print("start loading test data ...")
    TEST_DATASET = imtdataset(split='test', data_root=root, test_area=args.test_area)


    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True,collate_fn=collate_fn)

    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = IMT(out_feature=args.len_feature)
    model = MODEL.cuda()

    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    log_string('loaded model')

    '''Evaluate on chopped scenes'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        model = model.eval()
        results = []

        for i, (source, target, pose) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            source = source.data.numpy()
            target = target.data.numpy()
            pose_gt = pose.data.numpy()
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            pose_gt = torch.Tensor(pose)
            source, target, pose_gt= source.float().cuda(), target.float().cuda(), pose_gt.float().cuda()
            rot, tr = model(source, target)
            rot = rot.cpu().detach()
            tr = tr.cpu().detach()
            rot = rot.reshape(-1, 9)
            result = np.hstack((rot, tr))
            results.append(result)
    
    results = np.array(results, dtype=np.float32)
    np.savetxt(args.save_path + '/result1.txt', results.reshape(-1, 12))

if __name__ == '__main__':
    args = parse_args()
    main(args)