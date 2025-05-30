import torch
import numpy as np
from dataset import points_dataset
from utils import *
from model import PointTransformerLO38

def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """
    # print(np.asarray(list_data).shape)
    #
    # batch_sz = len(list_data)
    # print(batch_sz)

    point2 = [item[0] for item in list_data]
    # print(np.asarray(point2[0]).shape)
    point1 = [item[1] for item in list_data]

    sample_id = torch.from_numpy(np.asarray([item[2] for item in list_data]))
    T_gt = torch.from_numpy(np.asarray([item[3] for item in list_data]))
    T_trans = torch.from_numpy(np.asarray([item[4] for item in list_data]))
    T_trans_inv = torch.from_numpy(np.asarray([item[5] for item in list_data]))
    Tr = torch.from_numpy(np.asarray([item[6] for item in list_data]))


    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    # to_retain_as_list = []
    # data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    # data['T_gt'] = torch.stack([list_data[b]['T_gt'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)
    # data['T_trans'] = torch.stack([list_data[b]['T_trans'] for b in range(batch_sz)], dim=0) # (B, 4, 4)
    # data['T_trans_inv'] = torch.stack([list_data[b]['T_trans_inv'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)
    # data['Tr'] = torch.stack([list_data[b]['Tr'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)

    return point2, point1, sample_id, T_gt, T_trans, T_trans_inv, Tr

batch_size = 8

train_dataset = points_dataset(
    is_training = 1,
    num_point=150000,
    data_dir_list=[0, 1, 2, 3, 4, 5, 6],
    config=None
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_pair,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)#collate_fn=collate_pair,

for i, data in enumerate(train_loader):
    pos2, pos1, _, T_gt, T_trans, T_trans_inv, _ = data
    
    pos2 = [b.cuda() for b in pos2]
    pos1 = [b.cuda() for b in pos1]
    T_trans = T_trans.cuda().to(torch.float32)
    T_trans_inv = T_trans_inv.cuda().to(torch.float32)
    T_gt = T_gt.cuda().to(torch.float32)
    
    aug_frame = np.random.choice([1, 2], size = batch_size, replace = True) # random choose aug frame 1 or 2
    input_xyz_aug_f1, input_xyz_aug_f2, q_gt, t_gt = PreProcess(pos2, pos1, T_gt, T_trans, T_trans_inv, aug_frame)
    
    input_xyz_aug_f1_new = []
    input_xyz_aug_f2_new = []
    
    for i in range(len(input_xyz_aug_f1)):
        n1 = input_xyz_aug_f1[i].shape[0]
        n2 = input_xyz_aug_f2[i].shape[0]
        
        pos1 = np.zeros((150000, 3))
        pos2 = np.zeros((150000, 3))

        pos1[:n1, :3] = input_xyz_aug_f1[i].detach().cpu()[:, :3]
        pos2[:n2, :3] = input_xyz_aug_f2[i].detach().cpu()[:, :3]
        
        input_xyz_aug_f1_new.append(pos1)
        input_xyz_aug_f2_new.append(pos2)
    
    offset_f1_new = 0
    offset_f1_all = [0]
    for p in input_xyz_aug_f1_new:
        offset_f1_new += p.shape[0]
        offset_f1_all.append(offset_f1_new)
    
    offset_f2_new = 0
    offset_f2_all = [0]
    for p in input_xyz_aug_f2_new:
        offset_f2_new += p.shape[0]
        offset_f2_all.append(offset_f2_new)
    
    input_xyz_aug_f1_new = torch.vstack([torch.from_numpy(p) for p in input_xyz_aug_f1_new])
    input_xyz_aug_f2_new = torch.vstack([torch.from_numpy(p) for p in input_xyz_aug_f2_new])
    
    offset_f1_all = torch.tensor(offset_f1_all)
    offset_f2_all = torch.tensor(offset_f2_all)
    
    model = PointTransformerLO38().to("cuda:0")
    
    # print(input_xyz_aug_f1_new.shape)
    
    input_prev = (input_xyz_aug_f1_new.to("cuda:0").type(torch.float32), input_xyz_aug_f1_new.clone().to("cuda:0").type(torch.float32), offset_f1_all.to("cuda:0"))
    input_curr = (input_xyz_aug_f2_new.to("cuda:0").type(torch.float32), input_xyz_aug_f2_new.clone().to("cuda:0").type(torch.float32), offset_f2_all.to("cuda:0"))
    
    x = model(input_prev, input_curr)
    
    print("success")
    
    