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
- **[0]** Leverage function to read google sheet template timing + frame_idx

- **[1] Normalized Framing Algorithm:** Takes in general multi-K array ⇒ Bins to counts of 1,000

- **[2] Unpack function:** Loop through and optimally unpack all arrays 
    1) generate histograms, 
    2) generate unified data structure  for each video with helper [1]

- **[3] Unified Collation Function:** Given phase start-end times, return dictionary with total events across all phases 
    ⇒ Collate dictionaries to generate phase medians and standard deviations by leveraging [0]

- **[4] Unified Binning Function:** Returned 1K-length matrices with interquartile ranges across full range of bins for all events via using [2]
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

def plot_histogram(data_array, save_path):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(data_array)), data_array)
    plt.xlabel('Bin Index')
    plt.ylabel('Count')
    plt.title('Event Distribution')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def unpack_stats():
    video_dirs = os.listdir(master_save_dir)
    collated_save_dir = os.path.join(master_save_dir, "collated")
    if not os.path.exists(collated_save_dir):
        os.makedirs(collated_save_dir)

    # Hyperparameters 
    hp_dict = {
        "tool": {
            "wrist": [0.2, 0.3, 0.4, 0.5],
            "time": [10, 15, 20, 25]
        },
        "collide": {
            "radius": [0.25, 0.5, 0.75, 1]
        },
        "group_attn": {
            "mf": [10, 15, 20, 25],
            "thresh": [1, 2, 3, 4]
        },
        "attn": {
            "tf": [10, 15, 20, 25],
            "me": [5, 10, 15, 20]
        }
    }

    for video_dir in video_dirs:
        if video_dir == "collated":
            continue
            
        video_path = os.path.join(master_save_dir, video_dir)
        
        # Get all subdirectories
        metric_dirs = ["tool", "collide", "group_attn", "attn", "non_hp"]
        return_video_dict = {} 
        save_video_path = os.path.join(collated_save_dir, video_dir + ".pkl")
        
        for metric_dir in metric_dirs:
            dir_path = os.path.join(video_path, metric_dir)
            if not os.path.exists(dir_path):
                continue

            elif metric_dir == "group_attn":
                for mf in hp_dict["group_attn"]["mf"]:
                    for thresh in hp_dict["group_attn"]["thresh"]:
                        suffix = f"mf_{mf}_thresh_{thresh}"
                        file_path = os.path.join(dir_path, f"attn_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb"))
                        event_record = normalize_bins(data["event_record"])
                        unique_source_record = normalize_bins(data["unique_source_record"])
                        return_video_dict["group_attn_record"] = event_record
                        return_video_dict["group_attn_unique"] = unique_source_record

            elif metric_dir == "attn":
                for tf in hp_dict["attn"]["tf"]:
                    for me in hp_dict["attn"]["me"]:
                        suffix = f"tf_{tf}_me_{me}"
                        file_path = os.path.join(dir_path, f"attn_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb")) 
                        data = normalize_bins(data)
                        return_video_dict["attn_record"] = data

            elif metric_dir == "tool":
                for wrist in hp_dict["tool"]["wrist"]:
                    for time in hp_dict["tool"]["time"]:
                        suffix = f"wrist_{str(wrist).replace('.','')}_time_{time}"
                        file_path = os.path.join(dir_path, f"tool_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb")) 
                        data = normalize_bins(data)
                        return_video_dict["tool_record"] = data

            elif metric_dir == "collide":
                for radius in hp_dict["collide"]["radius"]:
                    suffix = f"radius_{str(radius).replace('.','')}"
                    file_path = os.path.join(dir_path, f"collide_{suffix}.pkl")
                    data = pickle.load(open(file_path, "rb")) 
                    data = normalize_bins(data)
                    return_video_dict["collide_record"] = data
                        
            elif metric_dir == "non_hp":
                non_hp_files = ["group_dispersion.pkl", "group_drift.pkl", "group_dist.pkl"]
                disp_data = pickle.load(open(os.path.join(dir_path, "group_dispersion.pkl"), "rb"))
                drift_data = pickle.load(open(os.path.join(dir_path, "group_drift.pkl"), "rb"))
                dist_data = pickle.load(open(os.path.join(dir_path, "group_dist.pkl"), "rb"))
                disp_data = normalize_bins(disp_data)
                drift_data = normalize_bins(drift_data)
                dist_data = normalize_bins(dist_data)
                return_video_dict["group_disp_record"] = disp_data
                return_video_dict["group_drift_record"] = drift_data
                return_video_dict["group_dist_record"] = dist_data
        pickle.dump(return_video_dict, open(save_video_path, "wb"))

def histogram_wrapper():
    hist_save_path = "/pasteur/u/bencliu/baseline/group_theory/empirical/histograms"
    video_dirs = os.listdir(master_save_dir)
    collated_save_dir = os.path.join(master_save_dir, "collated")

    # Hyperparameters 
    hp_dict = {
        "tool": {
            "wrist": [0.2, 0.3, 0.4, 0.5],
            "time": [10, 15, 20, 25, 50, 100]
        },
        "collide": {
            "radius": [0.25, 0.5, 0.75, 1]
        },
        "group_attn": {
            "mf": [10, 15, 20, 25],
            "thresh": [1, 2, 3, 4]
        },
        "attn": {
            "tf": [10, 15, 20, 25, 50, 100],
            "me": [5, 10, 15, 20]
        }
    }

    for video_dir in video_dirs:
        if video_dir == "collated":
            continue
            
        video_path = os.path.join(master_save_dir, video_dir)
        
        # Get all subdirectories
        metric_dirs = ["tool", "group_attn", "attn"] #"collide", "non_hp"
        return_video_dict = {} 
        dir_histogram_save_path = os.path.join(hist_save_path, video_dir)
        if not os.path.exists(dir_histogram_save_path):
            os.makedirs(dir_histogram_save_path)
        
        for dir_index, metric_dir in enumerate(metric_dirs):
            if dir_index == 5:
                print("DONE")
                breakpoint() 
            dir_path = os.path.join(video_path, metric_dir)
            if not os.path.exists(dir_path):
                continue

            if metric_dir == "tool":
                for wrist in hp_dict["tool"]["wrist"]:
                    for time in hp_dict["tool"]["time"]:
                        suffix = f"wrist_{str(wrist).replace('.','')}_time_{time}"
                        file_path = os.path.join(dir_path, f"tool_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb")) 
                        data = normalize_bins(data)
                        return_video_dict["tool_record"] = data
                        save_path = os.path.join(dir_histogram_save_path, f"tool_{suffix}.png")
                        plot_histogram(data, save_path)

            elif metric_dir == "group_attn":
                for mf in hp_dict["group_attn"]["mf"]:
                    for thresh in hp_dict["group_attn"]["thresh"]:
                        suffix = f"mf_{mf}_thresh_{thresh}"
                        file_path = os.path.join(dir_path, f"attn_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb"))
                        event_record = normalize_bins(data["event_record"])
                        unique_source_record = normalize_bins(data["unique_source_record"])
                        save_path = os.path.join(dir_histogram_save_path, f"group_attn_{suffix}.png")
                        plot_histogram(event_record, save_path)
                        save_path = os.path.join(dir_histogram_save_path, f"group_attn_unique_{suffix}.png")
                        plot_histogram(unique_source_record, save_path)

            elif metric_dir == "attn":
                for tf in hp_dict["attn"]["tf"]:
                    for me in hp_dict["attn"]["me"]:
                        suffix = f"tf_{tf}_me_{me}"
                        file_path = os.path.join(dir_path, f"attn_{suffix}.pkl")
                        data = pickle.load(open(file_path, "rb")) 
                        data = normalize_bins(data)
                        return_video_dict["attn_record"] = data
                        save_path = os.path.join(dir_histogram_save_path, f"attn_{suffix}.png")
                        plot_histogram(data, save_path)

            elif metric_dir == "collide":
                for radius in hp_dict["collide"]["radius"]:
                    suffix = f"radius_{str(radius).replace('.','')}"
                    file_path = os.path.join(dir_path, f"collide_{suffix}.pkl")
                    data = pickle.load(open(file_path, "rb")) 
                    data = normalize_bins(data)
                    return_video_dict["collide_record"] = data
                    save_path = os.path.join(dir_histogram_save_path, f"collide_{suffix}.png")
                    # plot_histogram(data, save_path)

            elif metric_dir == "non_hp":
                non_hp_files = ["group_dispersion.pkl", "group_drift.pkl", "group_dist.pkl"]
                disp_data = pickle.load(open(os.path.join(dir_path, "group_dispersion.pkl"), "rb"))
                drift_data = pickle.load(open(os.path.join(dir_path, "group_drift.pkl"), "rb"))
                dist_data = pickle.load(open(os.path.join(dir_path, "group_dist.pkl"), "rb"))
                disp_data = normalize_bins(disp_data)
                drift_data = normalize_bins(drift_data)
                dist_data = normalize_bins(dist_data)
                save_path = os.path.join(dir_histogram_save_path, f"group_dispersion.png")
                # plot_histogram(disp_data, save_path)
                save_path = os.path.join(dir_histogram_save_path, f"group_drift.png")
                # plot_histogram(drift_data, save_path)
                save_path = os.path.join(dir_histogram_save_path, f"group_dist.png")
                # plot_histogram(dist_data, save_path)

def group_analysis():
    hist_save_path = "/pasteur/u/bencliu/baseline/group_theory/empirical/histograms"
    video_dirs = os.listdir(master_save_dir)
    collated_save_dir = os.path.join(master_save_dir, "collated")

    for video_dir in video_dirs:
        if video_dir == "collated":
            continue
        video_path = os.path.join(master_save_dir, video_dir)
        
        # Get all subdirectories
        metric_dirs = ["non_hp", "collide"]
        return_video_dict = {} 
        
        for metric_dir in metric_dirs:
            dir_path = os.path.join(video_path, metric_dir)
            if not os.path.exists(dir_path):
                continue

            elif metric_dir == "collide":
                radius = 0.25 
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
            breakpoint() 
        
        # Associate data with video bounds 


def tester():
    histogram_wrapper() 

if __name__ == "__main__":
    tester() 
