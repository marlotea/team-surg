import numpy as np
import pickle
import os
import io
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from joints import (
    MAIN_JOINTS, 
    JOINT_NAMES_GNN_ABLATIONS as JOINT_NAMES, 
    JOINT_GROUPS, SPATIAL_EDGES)
from typing import List, Tuple
from util import read_pickle

class GNNDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        split: str, 
        exclude_groups: List[str] =[], 
        num_frames: int = 150
    ):
        super().__init__()
        self.dataset = read_pickle(dataset_path)[split][0]
        self.exclude_groups = exclude_groups
        self.num_frames = num_frames
    
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    
    def __getitem__(self, idx: int) -> Data:
        x, edge_list, y = self.prepare_dataset(idx)
        return Data(x=x, edge_index=edge_list, y=y)
        
    @staticmethod
    def reshape_joints(input_array: np.array) -> np.array:
        length = input_array.shape[0] if input_array.shape != (84,) else 1
        return input_array.reshape(length, 28, 3)


    @staticmethod
    def get_filtered_joint_list(exclude_groups: List[str]) -> List[str]:
        # Flatten group names into a set of excluded joint names
        excluded = set()
        for group in exclude_groups:
            excluded.update(JOINT_GROUPS[group])

        # Final joint list after filtering
        final_joint_list = [j for j in JOINT_NAMES if j not in excluded]
        return final_joint_list


    @staticmethod
    def filter_edges(final_joint_list: List[str]) -> List[str]:
        joint_set = set(final_joint_list)
        filtered_edges = [(a, b) for a, b in SPATIAL_EDGES if a in joint_set and b in joint_set]
        return filtered_edges


    @staticmethod
    def build_edge_list(joint_list: List[str], filtered_spatial_edges: List[str], num_frames=150) -> torch.Tensor:
        joint_idx = {name: i for i, name in enumerate(joint_list)}
        N = len(joint_list)
        total_nodes = N * num_frames

        rows, cols = [], []

        for t in range(num_frames):
            offset = t * N

            # spatial connections within the frame
            for a, b in filtered_spatial_edges:
                i, j = joint_idx[a] + offset, joint_idx[b] + offset
                rows += [i, j]
                cols += [j, i]

            # temporal connections between same joints across frames
            if t < num_frames - 1:
                next_offset = (t + 1) * N
                for i in range(N):
                    rows += [offset + i, next_offset + i]
                    cols += [next_offset + i, offset + i]

        return torch.tensor([rows, cols], dtype = torch.long)


    @staticmethod
    def build_node_list(exclude_groups : List, frames: np.array) -> torch.Tensor:
        joints_list = GNNDataset.get_filtered_joint_list(exclude_groups)
        joint_indices = [MAIN_JOINTS.index(joint) for joint in joints_list]
        reshaped_frames = GNNDataset.reshape_joints(frames)
        reshaped_frames = reshaped_frames[:, joint_indices]
        return torch.from_numpy(reshaped_frames.reshape(reshaped_frames.shape[0], reshaped_frames[1]*3))
    
    
    def prepare_dataset(self, idx: int) -> Tuple[torch.Tensor,torch.Tensor, float]:
        frames = self.dataset[idx]
        filtered_joint_list = GNNDataset.get_filtered_joint_list(self.exclude_groups)
        filtered_edges = GNNDataset.filter_edges(filtered_joint_list)
        edge_list = GNNDataset.build_edge_list(filtered_joint_list, filtered_edges, self.num_frames)
        x = GNNDataset.build_node_list(self.exclude_groups, frames)
        y = frames[-2] #action label 
        return x, edge_list, y


