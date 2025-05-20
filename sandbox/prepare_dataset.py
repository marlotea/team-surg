import numpy as np
import pickle
import os
import io
import torch
from joints import (
    MAIN_JOINTS, 
    JOINT_NAMES_GNN_ABLATIONS as JOINT_NAMES, 
    JOINT_GROUPS, SPATIAL_EDGES)
from typing import List


def reshape_joints(input_array: np.array) -> np.array:
    length = input_array.shape[0] if input_array.shape != (84,) else 1
    return input_array.reshape(length, 28, 3)

def get_filtered_joint_list(exclude_groups: List) -> List:
    # Flatten group names into a set of excluded joint names
    excluded = set()
    for group in exclude_groups:
        excluded.update(JOINT_GROUPS[group])

    # Final joint list after filtering
    final_joint_list = [j for j in JOINT_NAMES if j not in excluded]
    return final_joint_list

def filter_edges(final_joint_list: List) -> List:
    joint_set = set(final_joint_list)
    filtered_edges = [(a, b) for a, b in SPATIAL_EDGES if a in joint_set and b in joint_set]
    return filtered_edges

def build_edge_list(joint_list: List, filtered_spatial_edges: List, num_frames=150):
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

def build_node_list(exluded_groups : List, frames: np.array) -> np.array:
     joints_list = get_filtered_joint_list(exluded_groups)
     joint_indices = [MAIN_JOINTS.index(joint) for joint in joints_list]
     reshaped_frames = reshape_joints(frames)
     return reshaped_frames[:, joint_indices]
     