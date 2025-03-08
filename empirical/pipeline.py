from util import get_time_slice, preprocess, apply_transform 
import fire 
import os
from group_prox import calculate_distance
from group_dist import get_distance_diff
from attn import count_focused_attention_events
from group_attn import group_focused_attention
from collide import detect_collisions
from tool import detect_engagement_event
import pickle
import numpy as np
import time 
master_save_dir = "/pasteur/data/ghent_surg/empirical_results/"

def save_pickle(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

#HPs: Wrist, Event Time 
def tool_helper(dataset, video_code):
    tool_save_dir = os.path.join(master_save_dir, video_code, "tool")
    if not os.path.exists(tool_save_dir):
        os.makedirs(tool_save_dir)
    elbow_threshold = 0.7 

    #HPs: Wrist, Event Time 
    wrist_threshold = [0.2, 0.3, 0.4, 0.5]
    event_time_threshold = [10, 15, 20, 25, 50, 100]  
    for wrist_thresh in wrist_threshold:
        for event_time_thresh in event_time_threshold:
            suffix = f"wrist_{wrist_thresh}_time_{event_time_thresh}"
            suffix = suffix.replace(".", "")
            save_path = os.path.join(tool_save_dir, f"tool_{suffix}.pkl")
            # if os.path.exists(save_path):
            #     continue   
            __, __, event_record = detect_engagement_event(dataset, wrist_thresh, elbow_threshold, event_time_thresh)
            save_pickle(event_record, save_path)
            assert len(event_record) == len(dataset)

#HP: Radius
def collide_helper(dataset, video_code):
    collide_save_dir = os.path.join(master_save_dir, video_code, "collide")
    if not os.path.exists(collide_save_dir):
        os.makedirs(collide_save_dir)
    
    # Hyperparameters
    RADIUS = [0.25, 0.5, 0.75, 1]
    V_THRESH = 2
    for rad in RADIUS:
        suffix = f"radius_{rad}"
        suffix = suffix.replace(".", "")
        save_path = os.path.join(collide_save_dir, f"collide_{suffix}.pkl")
        if os.path.exists(save_path):
            continue  
        __, __, frame_collisions = detect_collisions(dataset,rad, V_THRESH) 
        save_pickle(frame_collisions, save_path)
        assert len(frame_collisions) == len(dataset)
    
#HP: MF, vec_dist 
def group_attn_helper(dataset, video_code):
    attn_save_dir = os.path.join(master_save_dir, video_code, "group_attn")
    if not os.path.exists(attn_save_dir):
        os.makedirs(attn_save_dir)
    
    data = [] 
    for frame in dataset:
        tracker_dict = {}
        for i, tracker_id in enumerate(frame['trackers']):
            person_joints = frame['joints3d'][i]
            tracker_dict[tracker_id] = person_joints
        data.append(tracker_dict)
    
    # Hyperparameters
    MIN_FRAMES_LIST = [10, 15, 20, 25] 
    DISTANCE_BETWEEN_VECTORS_THRESHOLD = [1, 2, 3, 4] 
    for min_frames in MIN_FRAMES_LIST:
        for distance_between_vectors_threshold in DISTANCE_BETWEEN_VECTORS_THRESHOLD:
            suffix = f"mf_{min_frames}_thresh_{distance_between_vectors_threshold}"
            save_path = os.path.join(attn_save_dir, f"attn_{suffix}.pkl")
            if os.path.exists(save_path):
                continue 
            group_focused_time, event_record, unique_source_record = group_focused_attention(data, min_frames, distance_between_vectors_threshold)
            save_dict = {"event_record": event_record, "unique_source_record": unique_source_record}
            save_pickle(save_dict, save_path)
            assert len(event_record) == len(dataset) == len(unique_source_record)

#HP: TF, ME 
def attn_helper(dataset, video_code):
    attn_save_dir = os.path.join(master_save_dir, video_code, "attn")
    if not os.path.exists(attn_save_dir):
        os.makedirs(attn_save_dir)
    data = {}
    start_frame_dict = {} 
    total_num_frames = len(dataset)
    step_size = 3 
    for frame_idx, data_i in enumerate(dataset):
        joints = data_i['joints3d']
        for i, tracker_id in enumerate(data_i['trackers']):
            person_joints = joints[i]
            if tracker_id in data:
                data[tracker_id].append(person_joints)
            else:
                data[tracker_id] = [person_joints]
                start_frame_dict[tracker_id] = frame_idx   

    #Hyperparams 
    time_frame = [10, 15, 20, 25, 50, 100] 
    margin_of_error = [5, 10, 15, 20] 
    for tf in time_frame:
        for me in margin_of_error:
            suffix = f"tf_{tf}_me_{me}"
            save_path = os.path.join(attn_save_dir, f"attn_{suffix}.pkl")
            # if os.path.exists(save_path):
            #     continue 
            _, event_record = count_focused_attention_events(data, me, tf, start_frame_dict, total_num_frames)
            save_pickle(event_record, save_path)
            assert len(event_record) == len(dataset)

#HP: None 
def group_prox_helper(dataset, video_code):
    save_path_disp = os.path.join(master_save_dir, video_code, "non_hp", "group_dispersion.pkl")
    save_path_drift = os.path.join(master_save_dir, video_code, "non_hp", "group_drift.pkl")
    if os.path.exists(save_path_disp) and os.path.exists(save_path_drift):
        return 
    data = []
    for data_i in dataset:
        joints = data_i['joints3d']
        one_frame = [joints[i, 0, (0, 2)] for i in range(joints.shape[0])]
        one_frame = np.array(one_frame)
        data.append(one_frame)
    data = np.array(data, dtype=object)
    centroid_dispersion, centroid_drift = calculate_distance(data)
    save_pickle(centroid_dispersion, save_path_disp)
    save_pickle(centroid_drift, save_path_drift)
    assert len(centroid_dispersion) == len(centroid_drift) == len(dataset)

#HP: None 
def group_dist_helper(dataset, video_code):
    save_path = os.path.join(master_save_dir, video_code, "non_hp", "group_dist.pkl")
    if os.path.exists(save_path):
        return 
    data = []
    trackers = []
    for frame in dataset:
        data.append(frame['joints3d'])
        trackers.append(frame['trackers'])
    distance_diff = get_distance_diff(data, trackers)
    save_pickle(distance_diff, save_path)
    assert len(distance_diff) == len(dataset)

def pipeline(start=0, end=10): 
    master_dir = "/pasteur/data/ghent_surg/full_hmr_outputs/"
    # Get only immediate directories, filter out files
    dir_list = [d for d in os.listdir(master_dir) 
                if os.path.isdir(os.path.join(master_dir, d))]
    dir_list = [os.path.join(master_dir, d) for d in dir_list if "test" not in d]
    dir_list = dir_list[start:end]
    for video_dir_path in dir_list:
        video_code = video_dir_path.split("/")[-1]
        save_dir = os.path.join(master_save_dir, video_code)
        non_hp_dir = os.path.join(master_save_dir, video_code, "non_hp")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(non_hp_dir):
            os.makedirs(non_hp_dir)
        
        # Load Dataset 
        time_start = time.time()
        input_folder_path = os.path.join(video_dir_path, "joint_out")
        dataset = get_time_slice(0, 0, input_folder_path, debug=False) 
        dataset = preprocess(dataset) 
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Frame-1: 
        time_start = time.time()
        tool_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Frame-2: 
        time_start = time.time()
        collide_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Frame-3: 
        time_start = time.time()
        distance_diff = group_dist_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Window-1: 
        time_start = time.time()
        group_attn_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Window-2: 
        time_start = time.time()
        group_prox = group_prox_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")

        #Window-3: 
        time_start = time.time()
        attn = attn_helper(dataset, video_code)
        time_end = time.time()
        print(f"Execution time: {(time_end - time_start) / 60} minutes")
        
        
if __name__ == "__main__": 
    fire.Fire()
