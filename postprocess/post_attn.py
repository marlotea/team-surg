import fire 
import os
import pickle
import numpy as np
import time 
from util import find_matching_mp4_files
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt

""" 
5. **Archived Notes:**
    1. First step, graph histograms to decide on hyper-params five example videos 
    2. Percentage changes of {Group attention events, Visual attention events, Tool events}
        1. Each PVI phase {Access, Lesion, Closure} compared to baseline levels
        2. Data Source: Core event definitions > Matrices
    3. Emergence of shared {Tool-Attn, Tool-GroupAttn} events across operative periods // PVI phases and compared to one another > Proxies for phase identification 
        1. {Access, Lesion, Closure - Surgical} compared to {Preparation, Check - Base}
        2. Data Source: Core event definitions > Matrices
    4. Long {Attention, Tooling} events occurrences > Comparison of attention types 
        1. Data Source: Enumeration of time duration for {Attention, Tooling} > Matrices 
    5. Comparison of transitory vs. in-phase events. 
        1. Data Source: Wrangle 
6. **Figures with Panels:** 
    1. 1) Example distribution of cumulative attention and tooling events for as single procedure
    2. 2) Multi-bar chart comparing short>long {attention, tooling} events with X-axis times and colors as operative periods 
    3. 3) PVI multi-bar chart with different events with percentage relative to baseline at each phase
    4. 4) Example image of multiple figures producing attention vectors (long triangle)
"""

def video_code_key_dict():
    mp4_files = find_matching_mp4_files()  
    video_dict = {}
    for mp4_file in mp4_files:
        video_code = mp4_file.split("/")[-2].replace(" ", "_")
        video_key = mp4_file.split("/")[-1]
        video_dict[video_key] = video_code  
    return video_dict 

# Global Vars 
video_dict = video_code_key_dict() 
master_save_dir = "/pasteur/data/ghent_surg/empirical_results/"

def get_frame_idx(label_str="00:00:00", fps=3):   
    # Add "00:" prefix if label_str is in MM:SS format
    if label_str.count(":") == 2:
        formatted_time = label_str
    else:
        formatted_time = "00:" + label_str
    hours, minutes, seconds = formatted_time.split(":") 
    frame_idx = int(hours) * 3600 * fps + int(minutes) * 60 * fps + int(seconds) * fps 
    print(label_str, formatted_time, frame_idx)
    return frame_idx 

def process_labels():
    labels = pd.read_csv("/pasteur/u/bencliu/baseline/group_theory/empirical/metadata/gs_labels.csv") 
    phases = range(1, 7) 

    #Create placeholder columns 
    for phase in phases:
        labels[f"{phase}_start_frame"] = 0
        labels[f"{phase}_end_frame"] = 0 
    labels["video_code"] = ""

    for index, row in tqdm(labels.iterrows(), total=len(labels), desc="Processing labels"):
        video_key = row["Video Key"]
        video_code = video_dict[video_key]
        for phase in phases:
            start_time = row[f"{phase}_Start"]
            end_time = row[f"{phase}_End"] 
            start_frame = get_frame_idx(start_time)
            end_frame = get_frame_idx(end_time) 
            labels.loc[index, f"{phase}_start_frame"] = start_frame
            labels.loc[index, f"{phase}_end_frame"] = end_frame
        labels.loc[index, "video_code"] = video_code 
    labels.to_csv("/pasteur/u/bencliu/baseline/group_theory/empirical/metadata/gs_labels_frames.csv", index=False) 

def normalize_bins(data_array, num_bins=1000):
    # Convert to numpy array if not already
    data_array = np.array(data_array)
    
    # Calculate how many original bins should be combined into each new bin
    bin_size = len(data_array) / num_bins
    
    # Create the new array by summing groups of elements
    normalized_array = np.zeros(num_bins)
    for i in range(num_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        normalized_array[i] = np.sum(data_array[start_idx:end_idx])
    
    return normalized_array

def get_total_num_frames(video_code):
    image_save_dir = "/pasteur/data/ghent_surg/full_image_data"
    video_dir = os.path.join(image_save_dir, video_code)
    return len(os.listdir(video_dir))

def group_analysis():
    labels_df = pd.read_csv("/pasteur/u/bencliu/baseline/group_theory/empirical/metadata/gs_labels_frames.csv")
    hist_save_path = "/pasteur/u/bencliu/baseline/group_theory/empirical/histograms"
    video_dirs = os.listdir(master_save_dir)
    video_dirs = [dir for dir in video_dirs if dir in labels_df['video_code'].values and dir != "collated"]

    # Initialize master matrices
    master_metrics_matrix = []
    master_metrics_broad_matrix = []

    for video_code in video_dirs:
        video_path = os.path.join(master_save_dir, video_code)
        
        # Get all subdirectories
        metric_dirs = ["non_hp", "collide"]
        return_video_dict = {} 
        total_num_frames = get_total_num_frames(video_code)
        
        for metric_dir in metric_dirs:
            dir_path = os.path.join(video_path, metric_dir)
            if not os.path.exists(dir_path):
                continue

            elif metric_dir == "collide":
                radius = 1
                suffix = f"radius_{str(radius).replace('.','')}"
                file_path = os.path.join(dir_path, f"collide_{suffix}.pkl")
                data = pickle.load(open(file_path, "rb")) 
                collide_data = normalize_bins(data)

            elif metric_dir == "non_hp":
                non_hp_files = ["group_dispersion.pkl", "group_drift.pkl", "group_dist.pkl"]
                disp_data = pickle.load(open(os.path.join(dir_path, "group_dispersion.pkl"), "rb"))
                drift_data = pickle.load(open(os.path.join(dir_path, "group_drift.pkl"), "rb"))
                dist_data = pickle.load(open(os.path.join(dir_path, "group_dist.pkl"), "rb"))
                disp_data = normalize_bins(disp_data)
                drift_data = normalize_bins(drift_data)
                dist_data = normalize_bins(dist_data)
        
        df = labels_df[labels_df["video_code"] == video_code]

        # Calculate individual phase metrics 
        phase_indices = range(1, 7)
        metrics_matrix = np.zeros((6, 3))
        for phase in phase_indices:
            # Get start and end frames for this phase
            start_frame = df[f"{phase}_start_frame"].iloc[0]
            end_frame = df[f"{phase}_end_frame"].iloc[0] if phase < 6 else total_num_frames
            
            # Convert frames to normalized indices (0-1000)
            start_idx = max(0, int((start_frame / total_num_frames) * 1000))
            end_idx = min(999, int((end_frame / total_num_frames) * 1000))
            num_bins = end_idx - start_idx
            
            if num_bins <= 0:
                num_bins = 1 
                
            # Calculate average counts for each metric over the phase
            collide_sum = np.sum(collide_data[start_idx:end_idx])
            dist_sum = np.sum(dist_data[start_idx:end_idx])
            disp_sum = np.sum(disp_data[start_idx:end_idx])
            drift_sum = np.sum(drift_data[start_idx:end_idx])
            metrics_matrix[phase-1, 0] = collide_sum / num_bins
            metrics_matrix[phase-1, 1] = dist_sum / num_bins
            metrics_matrix[phase-1, 2] = disp_sum / num_bins

        # Calculate broad phase metrics
        # Preop (phases 1-2) | Operative (phases 3-5) | Postop (phase 6) |Transitions (10 bins ~ 1 minute)
        metrics_broad_matrix = np.zeros((4, 3))
        metrics_broad_matrix[0] = np.mean(metrics_matrix[0:2], axis=0)
        metrics_broad_matrix[1] = np.mean(metrics_matrix[2:5], axis=0)
        metrics_broad_matrix[2] = metrics_matrix[5]
        transition_sums = np.zeros(3)
        for phase in phase_indices:
            end_frame = df[f"{phase}_end_frame"].iloc[0] if phase < 6 else total_num_frames
            end_idx = min(989, int((end_frame / total_num_frames) * 1000))  # Ensure we don't go past array bounds
            
            # Sum up 10 bins after each phase end
            if not phase == 6:
                transition_sums[0] += np.sum(collide_data[end_idx:end_idx+20])
                transition_sums[1] += np.sum(dist_data[end_idx:end_idx+20])
                transition_sums[2] += np.sum(disp_data[end_idx:end_idx+20])
        
        # Average over all transitions (6 phases = 6 transitions)
        metrics_broad_matrix[3] = transition_sums / (6 * 10)  # Divide by total number of bins considered

        # Append matrices to master lists
        master_metrics_matrix.append(metrics_matrix)
        master_metrics_broad_matrix.append(metrics_broad_matrix)
    
    # Convert to numpy arrays
    master_metrics_matrix = np.array(master_metrics_matrix)
    master_metrics_broad_matrix = np.array(master_metrics_broad_matrix)
    
    # Calculate means, standard deviations, and IQ ranges
    metrics_mean = np.mean(master_metrics_matrix, axis=0)
    metrics_std = np.std(master_metrics_matrix, axis=0)
    metrics_q25 = np.percentile(master_metrics_matrix, 25, axis=0)
    metrics_q75 = np.percentile(master_metrics_matrix, 75, axis=0)
    metrics_iqr = metrics_q75 - metrics_q25

    metrics_broad_mean = np.mean(master_metrics_broad_matrix, axis=0)
    metrics_broad_std = np.std(master_metrics_broad_matrix, axis=0)
    metrics_broad_q25 = np.percentile(master_metrics_broad_matrix, 25, axis=0)
    metrics_broad_q75 = np.percentile(master_metrics_broad_matrix, 75, axis=0)
    metrics_broad_iqr = metrics_broad_q75 - metrics_broad_q25
    
    # Create visualizations
    plot_phase_metrics(metrics_mean, metrics_std, name="mean")
    plot_broad_metrics(metrics_broad_mean, metrics_broad_std, name="mean")
    plot_phase_metrics(metrics_mean, metrics_std, metrics_q25, metrics_q75, name="median")
    plot_broad_metrics(metrics_broad_mean, metrics_broad_std, metrics_broad_q25, metrics_broad_q75, name="median")
    breakpoint() 

def plot_phase_metrics(means, stds, q25=None, q75=None, name="median", error_bars=False):
    phases = ['Preparation', 'Draping', 'Access', 'Lesion Treatment', 'Closure', 'Check']
    metrics = ['No. Collisions', 'Distance Traversal', 'Relative Dispersion']
    
    x = np.arange(len(phases))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    for i in range(len(metrics)):
        if q25 is not None and q75 is not None:
            medians = (q75[:, i] + q25[:, i])/2
            iqr = q75[:, i] - q25[:, i]
            bars = ax.bar(x + i*width - (len(metrics)*width/2), medians, width,
                       label=metrics[i], color=colors[i])
        else:
            bars = ax.bar(x + i*width - (len(metrics)*width/2), means[:, i], width, 
                       label=metrics[i], color=colors[i])
            if error_bars:
                ax.errorbar(x + i*width - (len(metrics)*width/2), means[:, i], 
                          yerr=stds[:, i], fmt='none', color='black', capsize=5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Normalized Measurements', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
             ncol=len(metrics), fontsize=12)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f'/pasteur/u/bencliu/baseline/group_theory/empirical/a1_group/a1_phase_distributions_{name}.png',
                bbox_inches='tight')
    plt.close()

def plot_broad_metrics(means, stds, q25=None, q75=None, name="median", error_bars=False):
    phases = ['Pre-operative', 'Operative', 'Post-operative', 'Transitory']
    metrics = ['No. Collisions', 'Distance Traversal', 'Relative Dispersion']
    
    x = np.arange(len(phases))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    for i in range(len(metrics)):
        if q25 is not None and q75 is not None:
            medians = (q75[:, i] + q25[:, i])/2
            iqr = q75[:, i] - q25[:, i]
            bars = ax.bar(x + i*width - (len(metrics)*width/2), medians, width,
                       label=metrics[i], color=colors[i])
        else:
            bars = ax.bar(x + i*width - (len(metrics)*width/2), means[:, i], width, 
                       label=metrics[i], color=colors[i])
            if error_bars:
                ax.errorbar(x + i*width - (len(metrics)*width/2), means[:, i], 
                          yerr=stds[:, i], fmt='none', color='black', capsize=5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Normalized Measurements', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
             ncol=len(metrics), fontsize=12)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f'/pasteur/u/bencliu/baseline/group_theory/empirical/a1_group/a1_broad_distributions_{name}.png',
                bbox_inches='tight')
    plt.close()

def tester():
    group_analysis() 

if __name__ == "__main__":
    tester() 
