"""
Author: Ljy
Date: Aug 2022
"""

import os
import numpy as np

import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader

def collate_fn(batch):
    batch_len = len(batch)
    source_batch = []
    target_batch = []
    pose_batch = []
    for i in range(batch_len):
        dic = batch[i]
        source_batch.append(dic[0])
        target_batch.append(dic[1])
        pose_batch.append(dic[2])
    res=[]
    res.append(pad_sequence(source_batch,batch_first=True))
    res.append(pad_sequence(target_batch,batch_first=True))
    res.append(pad_sequence(pose_batch,batch_first=True))
    return res

def collate_fn_geo(batch):
    batch_len = len(batch)
    source_batch = []
    target_batch = []
    pose_batch = []
    source_point = []
    for i in range(batch_len):
        dic = batch[i]
        source_batch.append(dic[0])
        target_batch.append(dic[1])
        pose_batch.append(dic[2])
        source_point.append(dic[3])
    res=[]
    res.append(pad_sequence(source_batch,batch_first=True))
    res.append(pad_sequence(target_batch,batch_first=True))
    res.append(pad_sequence(pose_batch,batch_first=True))
    res.append(pad_sequence(source_point, batch_first=True))
    return res

class imtdataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea/',  test_area=0):
        super().__init__()
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.source_data = []
        self.target_data = []
        self.pose = []
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            source_path = room_path + '/source_rt/'
            target_path = room_path + '/target_rt/'
            pose_path = room_path + '/pose_rt.txt'
            pose = np.loadtxt(pose_path)
            source_files = os.listdir(source_path)
            target_files = os.listdir(target_path)
            source_files.sort(key=lambda x: int(x[:6]))
            target_files.sort(key=lambda x: int(x[:6]))
            for j in range(len(source_files)):
                source = np.fromfile(source_path+source_files[j], dtype=np.float32).reshape(-1, 7)
                target = np.fromfile(target_path+target_files[j], dtype=np.float32).reshape(-1, 7)
                self.source_data.append(source)
                self.target_data.append(target)
                self.pose.append(pose[j])
        print("Totally {} samples in {} set.".format(len(self.pose), split))

    def __getitem__(self, idx):
        source = torch.tensor(self.source_data[idx], dtype=torch.float32)
        target = torch.tensor(self.target_data[idx], dtype=torch.float32)
        pose = torch.tensor(self.pose[idx], dtype=torch.float32)
        return source, target, pose

    def __len__(self):
        return len(self.pose)


class imtdataset_geo(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea/',  test_area=0):
        super().__init__()
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.source_data = []
        self.target_data = []
        self.pose = []
        self.source_point = []
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            source_path = room_path + '/source_rt/'
            target_path = room_path + '/target_rt/'
            pose_path = room_path + '/pose_rt.txt'
            source_point = room_path + '/source_pt/'
            pose = np.loadtxt(pose_path)
            source_files = os.listdir(source_path)
            target_files = os.listdir(target_path)
            source_pt_files = os.listdir(source_path)
            source_files.sort(key=lambda x: int(x[:6]))
            target_files.sort(key=lambda x: int(x[:6]))
            source_pt_files.sort(key=lambda x: int(x[:6]))
            for j in range(len(source_files)):
                source = np.fromfile(source_path+source_files[j], dtype=np.float32).reshape(-1, 7)
                target = np.fromfile(target_path+target_files[j], dtype=np.float32).reshape(-1, 7)
                source_pt = np.fromfile(source_point+source_pt_files[j], dtype=np.float32).reshape(-1, 4)
                self.source_data.append(source)
                self.target_data.append(target)
                self.pose.append(pose[j])
                self.source_point.append(source_pt[:, :3])
        print("Totally {} samples in {} set.".format(len(self.pose), split))

    def __getitem__(self, idx):
        source = torch.tensor(self.source_data[idx], dtype=torch.float32)
        target = torch.tensor(self.target_data[idx], dtype=torch.float32)
        pose = torch.tensor(self.pose[idx], dtype=torch.float32)
        source_point = torch.tensor(self.source_point[idx], dtype=torch.float32)
        return source, target, pose, source_point

    def __len__(self):
        return len(self.pose)

if __name__ == '__main__':
    # data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    # point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    # print('point data size:', point_data.__len__())
    # print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    # print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    # import torch, time, random
    # manual_seed = 123
    # random.seed(manual_seed)
    # np.random.seed(manual_seed)
    # torch.manual_seed(manual_seed)
    # torch.cuda.manual_seed_all(manual_seed)
    # def worker_init_fn(worker_id):
    #     random.seed(manual_seed + worker_id)
    # train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()


    root = '/media/puzek/sda/dataset/imt_data/dataset/sequences/'
    a = imtdataset(data_root=root)
    dataloader = DataLoader(dataset=a, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)
    print(dataloader)
    for step, (x1, x2, pose) in enumerate(dataloader):
        # print(step, x1, x2, pose)
        # print('step', step)
        print('*****************************')
        # print(x1.shape, pose.shape)
    print(a.__len__())
    print("^ ^")