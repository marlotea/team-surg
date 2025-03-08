import fire 
import os
import pickle
import numpy as np
import time 
from util import find_matching_mp4_files
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt

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

    # Initialize matrices for each metric
    master_metrics_matrix = []  # Phase-wise metrics
    master_metrics_broad_matrix = []  # Broad phase metrics

    # Add new matrices for attention and tool duration metrics
    master_attn_matrix = []
    master_attn_broad_matrix = []
    master_tool_matrix = []
    master_tool_broad_matrix = []

    for video_code in video_dirs:
        video_path = os.path.join(master_save_dir, video_code)
        total_num_frames = get_total_num_frames(video_code)
        metrics_data = {}
        tools_data = {} 
        attn_data = {}

        # Load tool duration data 
        tool_tfs = [10, 25, 50, 100] 
        for tf in tool_tfs:
            tool_path = os.path.join(video_path, "tool", f"tool_wrist_03_time_{tf}.pkl")
            if os.path.exists(tool_path):
                tool_data = pickle.load(open(tool_path, "rb"))
                tool_data = normalize_bins(tool_data)
                tools_data[str(tf)] = tool_data
        
        # Load attention duration data 
        attn_tfs = [10, 25, 50, 100] 
        for tf in attn_tfs:
            attn_path = os.path.join(video_path, "attn", f"attn_tf_{tf}_me_10.pkl")
            if os.path.exists(attn_path):
                attn_data[str(tf)] = normalize_bins(pickle.load(open(attn_path, "rb")))

        # Load attention (tf_50, me_10)
        attn_path = os.path.join(video_path, "attn", "attn_tf_50_me_10.pkl")
        metrics_data['attn'] = normalize_bins(pickle.load(open(attn_path, "rb")))

        # Load group attention (mf_25, thresh_1)
        group_attn_path = os.path.join(video_path, "group_attn", "attn_mf_25_thresh_1.pkl")
        if os.path.exists(group_attn_path):
            data = pickle.load(open(group_attn_path, "rb"))
            metrics_data['group_attn'] = normalize_bins(data['event_record'])
            metrics_data['group_attn_unique'] = normalize_bins(data['unique_source_record'])

        # Load tool (wrist_03, time_100)
        tool_path = os.path.join(video_path, "tool", "tool_wrist_03_time_100.pkl")
        if os.path.exists(tool_path):
            metrics_data['tool'] = normalize_bins(pickle.load(open(tool_path, "rb")))

        df = labels_df[labels_df["video_code"] == video_code]
        
        # Calculate individual phase metrics
        phase_indices = range(1, 7)
        metrics_matrix = np.zeros((6, len(metrics_data)))
        for phase in phase_indices:
            start_frame = df[f"{phase}_start_frame"].iloc[0]
            end_frame = df[f"{phase}_end_frame"].iloc[0] if phase < 6 else total_num_frames
            start_idx = max(0, int((start_frame / total_num_frames) * 1000))
            end_idx = min(999, int((end_frame / total_num_frames) * 1000))
            num_bins = max(1, end_idx - start_idx)
            if num_bins <= 0:
                num_bins = 1 
            
            for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
                phase_sum = np.sum(metric_data[start_idx:end_idx])
                metrics_matrix[phase-1, i] = phase_sum / num_bins
                
                # Calculate baseline normalized version
                baseline = np.mean(metric_data)
                if baseline > 0:
                    metrics_matrix[phase-1, i] /= baseline
        
        # Tool Matrix 
        tool_matrix = np.zeros((6, len(tools_data)))
        for phase in phase_indices:
            start_frame = df[f"{phase}_start_frame"].iloc[0]
            end_frame = df[f"{phase}_end_frame"].iloc[0] if phase < 6 else total_num_frames
            start_idx = max(0, int((start_frame / total_num_frames) * 1000))
            end_idx = min(999, int((end_frame / total_num_frames) * 1000))
            num_bins = max(1, end_idx - start_idx)
            if num_bins <= 0:
                num_bins = 1
            
            for i, (tf, tool_data) in enumerate(tools_data.items()):
                phase_sum = np.sum(tool_data[start_idx:end_idx])
                tool_matrix[phase-1, i] = phase_sum / num_bins
                baseline = np.mean(tool_data)
                if baseline > 0:
                    tool_matrix[phase-1, i] /= baseline
        
        # Attn Matrix 
        attn_matrix = np.zeros((6, len(attn_data)))
        for phase in phase_indices:
            start_frame = df[f"{phase}_start_frame"].iloc[0]
            end_frame = df[f"{phase}_end_frame"].iloc[0] if phase < 6 else total_num_frames
            start_idx = max(0, int((start_frame / total_num_frames) * 1000))
            end_idx = min(999, int((end_frame / total_num_frames) * 1000))
            num_bins = max(1, end_idx - start_idx)
            if num_bins <= 0:
                num_bins = 1 
            
            for i, (tf, attn_data_i) in enumerate(attn_data.items()):
                phase_sum = np.sum(attn_data_i[start_idx:end_idx])
                attn_matrix[phase-1, i] = phase_sum / num_bins
                baseline = np.mean(attn_data_i)
                if baseline > 0:
                    attn_matrix[phase-1, i] /= baseline

        # Calculate broad phase metrics (removing transitory)
        metrics_broad_matrix = np.zeros((3, len(metrics_data)))  # Changed from 4 to 3
        metrics_broad_matrix[0] = np.mean(metrics_matrix[0:2], axis=0)  # Preop
        metrics_broad_matrix[1] = np.mean(metrics_matrix[2:5], axis=0)  # Operative
        metrics_broad_matrix[2] = metrics_matrix[5]  # Postop
        master_metrics_broad_matrix.append(metrics_broad_matrix)
        master_metrics_matrix.append(metrics_matrix)

        # Tool Broad Matrix 
        tool_broad_matrix = np.zeros((3, len(tools_data)))
        tool_broad_matrix[0] = np.mean(tool_matrix[0:2], axis=0)
        tool_broad_matrix[1] = np.mean(tool_matrix[2:5], axis=0)
        tool_broad_matrix[2] = tool_matrix[5]
        master_tool_broad_matrix.append(tool_broad_matrix)
        master_tool_matrix.append(tool_matrix)

        # Attn Broad Matrix 
        attn_broad_matrix = np.zeros((3, len(attn_data)))
        attn_broad_matrix[0] = np.mean(attn_matrix[0:2], axis=0)
        attn_broad_matrix[1] = np.mean(attn_matrix[2:5], axis=0)
        attn_broad_matrix[2] = attn_matrix[5]
        master_attn_broad_matrix.append(attn_broad_matrix)
        master_attn_matrix.append(attn_matrix)

    # Process and plot results
    master_metrics_matrix = np.array(master_metrics_matrix)
    master_metrics_broad_matrix = np.array(master_metrics_broad_matrix)
    master_attn_matrix = np.array(master_attn_matrix)
    master_attn_broad_matrix = np.array(master_attn_broad_matrix)
    master_tool_matrix = np.array(master_tool_matrix)
    master_tool_broad_matrix = np.array(master_tool_broad_matrix)
    
    # Calculate statistics
    metrics_mean = np.mean(master_metrics_matrix, axis=0)
    attn_mean = np.mean(master_attn_matrix, axis=0)
    tool_mean = np.mean(master_tool_matrix, axis=0)
    metrics_std = np.std(master_metrics_matrix, axis=0)
    metrics_q25 = np.percentile(master_metrics_matrix, 25, axis=0)
    metrics_q75 = np.percentile(master_metrics_matrix, 75, axis=0)
    
    metrics_broad_mean = np.mean(master_metrics_broad_matrix, axis=0)
    attn_broad_mean = np.mean(master_attn_broad_matrix, axis=0)
    tool_broad_mean = np.mean(master_tool_broad_matrix, axis=0)
    metrics_broad_std = np.std(master_metrics_broad_matrix, axis=0)
    metrics_broad_q25 = np.percentile(master_metrics_broad_matrix, 25, axis=0)
    metrics_broad_q75 = np.percentile(master_metrics_broad_matrix, 75, axis=0)
    
    # Create visualizations
    plot_phase_metrics(metrics_mean, metrics_std, name="mean")
    plot_broad_metrics(metrics_broad_mean, metrics_broad_std, name="mean")
    plot_phase_metrics(metrics_mean, metrics_std, metrics_q25, metrics_q75, name="median")
    plot_broad_metrics(metrics_broad_mean, metrics_broad_std, metrics_broad_q25, metrics_broad_q75, name="median")
    
    # Create tool/attn visualizations 
    plot_duration_metrics(tool_broad_mean, "tool", "/pasteur/u/bencliu/baseline/group_theory/empirical/a2_attn/")
    plot_duration_metrics(attn_broad_mean, "attention", "/pasteur/u/bencliu/baseline/group_theory/empirical/a2_attn/")
    breakpoint() 

   

def plot_phase_metrics(means, stds, q25=None, q75=None, name="median", error_bars=False):
    phases = ['Preparation', 'Draping', 'Access', 'Lesion Treatment', 'Closure', 'Check']
    metrics = ['Attention Event', 'Group Attention Event', 'No. Attentive Groups', 'Hand-Tool Interaction']  
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    x = np.arange(len(phases))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
                          yerr=stds[:, i], fmt='none', color='black', capsize=3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Normalized Counts / Global Average', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
             ncol=len(metrics), fontsize=12)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f'/pasteur/u/bencliu/baseline/group_theory/empirical/a2_attn/a2_phase_distributions_{name}.png',
                bbox_inches='tight')
    plt.close()

def plot_broad_metrics(means, stds, q25=None, q75=None, name="median", error_bars=False):
    phases = ['Pre-operative', 'Operative', 'Post-operative']
    metrics = ['Attention Event', 'Group Attention Event', 'No. Attentive Groups', 'Hand-Tool Interaction']  
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    x = np.arange(len(phases))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(len(metrics)):
        if q25 is not None and q75 is not None:
            medians = (q75[:, i] + q25[:, i])/2
            iqr = q75[:, i] - q25[:, i]
            bars = ax.bar(x + i*width - (len(metrics)*width/2), medians, width, 
                         label=metrics[i], color=colors[i])
        else:
            bars = ax.bar(x + i*width - (len(metrics)*width/2), means[:, i], width, 
                         label=metrics[i], color=colors[i])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Normalized Counts / Global Average', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
             ncol=len(metrics), fontsize=12)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f'/pasteur/u/bencliu/baseline/group_theory/empirical/a2_attn/a2_broad_distributions_{name}.png',
                bbox_inches='tight')
    plt.close()

def plot_duration_metrics(broad_mean, metric_type, save_dir):
    """
    Plot duration-based metrics with operative phases as bar colors.
    Args:
        broad_mean: numpy array of shape (3, num_time_settings)
        metric_type: string, either 'attention' or 'tool'
        save_dir: path to save the plot
    """
    phases = ['Pre-operative', 'Operative', 'Post-operative']
    time_settings = [5, 10, 15, 30] # [10, 25, 50, 100] ~ frame counts => Round up to nearest 5th second
    colors = plt.cm.viridis(np.linspace(0, 1, len(phases)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(time_settings))
    width = 0.25  # Width of bars
    
    # Plot bars for each operative phase
    for i, phase in enumerate(phases):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, broad_mean[i], width, 
                     label=phase, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Customize plot
    suffix = "Attention Event" if metric_type == "attention" else "Tool Event"
    ax.set_ylabel(f'Normalized {suffix} Counts / Global Average', fontsize=12)
    ax.set_xlabel('Time Window (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(time_settings, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Place legend directly above plot without extra space
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
             ncol=len(phases), fontsize=12)
    
    # Set style
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    metric_name = 'attention' if metric_type == 'attention' else 'tool'
    plt.savefig(os.path.join(save_dir, f'duration_{metric_name}_phases.png'),
                bbox_inches='tight')
    plt.close()

def tester():
    group_analysis() 

if __name__ == "__main__":
    tester() 
