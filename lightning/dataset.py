import pandas as pd 
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from torch.utils.data import IterableDataset, get_worker_info

import copy
from skspatial.objects import Point, Vector
import pickle5 as pickle 
import matplotlib.colors as mcolors
from tqdm import tqdm 
from joints import (pelvic_indices, arm_indices, head_indices, thorax_indices, leg_indices, spine_indices,
                    pelvic_indices_pose, arm_indices_pose, head_indices_pose, thorax_indices_pose, leg_indices_pose, spine_indices_pose,
                    head_struct_indices, eye_indices, ear_indices, elbow_indices, wrist_indices, SMPL_JOINT_NAMES, JOINT_NAMES) 

import torch
from PIL import Image
import torchvision.transforms as T
import random
from downstream import * 

class MixerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split):
        self.dataset = read_pickle(dataset)[split]
    
    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):
        dataset_example = self.dataset[index] 
        output = {} 
        output['label'] = dataset_example[1] 
        output['embedding_seq'] = dataset_example[0] 
        return output  
             
    """
    Paramters: first_phase, second_phase (Phases being classified)
               time: time in seconds
    Returns: list of training examples, where num_frames per example determined by time
    """            
    
    
class IterativeDataset(IterableDataset):
    def __init__(self, file_path, split="train"):
        self.file_path = file_path
        self.split = split

    def _load_data(self):
        return read_pickle(self.file_path)[self.split]

    def __iter__(self):
        data = list(self._load_data())  # Convert to list for indexing

        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker: return full dataset
            for item in data:
                x = torch.tensor(item["data"], dtype=torch.float32)
                y = torch.tensor(item["label"], dtype=torch.long)
                yield x, y
        else:
            # Multiple workers: split dataset
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            chunk_size = len(data) // num_workers
            start = worker_id * chunk_size
            end = start + chunk_size if worker_id < num_workers - 1 else len(data)

            # Yield only the subset of data for this worker
            for item in data[start:end]:
                x = torch.tensor(item["data"], dtype=torch.float32)
                y = torch.tensor(item["label"], dtype=torch.long)
                yield x, y


def generate_dictionaries(labels, examples):
    output_dict = {}
    for i in range(len(labels)):
        if labels[i] not in output_dict:
            output_dict[labels[i]] = np.array([examples[i]])
        else:
            output_dict[labels[i]]= np.append(output_dict[labels[i]], np.array([examples[i]]), axis = 0)
    return output_dict         

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')
        
#Gnerate training splits 
def generate_splits(total_length):
    train_size = int(0.7 * total_length)
    valid_size = int(0.15 * total_length)
    test_size = total_length - train_size - valid_size

    # Create a list of labels representing the split
    labels = ["train"] * train_size + ["valid"] * valid_size + ["test"] * test_size

    # Shuffle the labels to randomize the split
    random.shuffle(labels)
    return labels 

def shuffle_dict(input): 
    keys = list(input.keys())
    random.shuffle(keys)
    return {key: input[key] for key in keys}

def apply_transform(data, flip_yz=False): 
    vertices = (data['vertices'] + data['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    joints = (data['joints3d'] + data['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    if flip_yz:
        vertices = vertices[..., [0, 2, 1]]
        joints = joints[..., [0, 2, 1]]
    return joints, vertices 


def apply_padding(frame_at_joints3d, p_max = 11, j = 127):
    p_actual = frame_at_joints3d.shape[0]
    
    # Move tensor to CPU if it's on CUDA
    if isinstance(frame_at_joints3d, torch.Tensor):
        frame_at_joints3d = frame_at_joints3d.detach().cpu().numpy()

    padded_tensor = np.zeros((p_max, j, 3))  # Initialize numpy array
    padded_tensor[:p_actual, ...] = frame_at_joints3d  # Copy data

    return padded_tensor.astype(np.float32)  # Ensure proper dtype
    
