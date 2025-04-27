import subprocess

seq_len_ablation_commands = [
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_150.pkl --seq_len 150 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_150",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_125.pkl --seq_len 125 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_125",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_100.pkl --seq_len 100 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_100",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_75.pkl --seq_len 75 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_75",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_50.pkl --seq_len 50 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_50",
]

seq_len_two_commands = [
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_25.pkl --seq_len 25 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_25",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_10.pkl --seq_len 10 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_10",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_5.pkl --seq_len 5 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_5",
]

joint_classes_commands = [
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_pelvis_sampled_3.pkl --seq_len 50 --embedd_dim 9 --batch_size 32 --exp_name abl_param_pelvis_weighted_bs32", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_arm_sampled_3.pkl --seq_len 50 --embedd_dim 21 --batch_size 32 --exp_name abl_param_pelvis_arm_weighted_bs32", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_head_sampled_3.pkl --seq_len 50 --embedd_dim 42 --batch_size 32 --exp_name abl_param_pelvis_arm_head_weighted_bs32", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_thorax_sampled_3.pkl --seq_len 50 --embedd_dim 57 --batch_size 32 --exp_name abl_param_pelvis_arm_head_thorax_weighted_bs32", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_spine_sampled_3.pkl --seq_len 50 --embedd_dim 66 --batch_size 32 --exp_name abl_param_pelvis_arm_head_thorax_spine_weighted_bs32", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_3.pkl --seq_len 50 --embedd_dim 84 --batch_size 32 --exp_name abl_param_pelvis_arm_head_thorax_spine_leg_weighted_bs32", 
]

joint_indiv_commands = [
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_vision_arm_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 30 --exp_name abl_ijoints_vision_arm",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_vision_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 18 --exp_name abl_ijoints_vision",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_arm_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 12 --exp_name abl_ijoints_arm",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_wrists_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 6 --exp_name abl_ijoints_wrists",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_wrists_elbows_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 12 --exp_name abl_ijoints_wrists_elbows", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_wrists_elbows_eyes_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 18 --exp_name abl_ijoints_wrists_elbows_eyes",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_wrists_elbows_eyes_head_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 24 --exp_name abl_ijoints_wrists_elbows_eyes_head",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_wrists_elbows_eyes_head_ear_sampled_3.pkl --batch_size 32 --seq_len 50 --embedd_dim 30 --exp_name abl_ijoints_wrists_elbows_eyes_head_ear",
]

pose_commands = [
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_pelvis_sampled_3.pkl --seq_len 50 --embedd_dim 27 --batch_size 32 --exp_name abl_param_pose_pelvis", 
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_arm_sampled_3.pkl --seq_len 50 --embedd_dim 63 --batch_size 32 --exp_name abl_param_pose_pelvis_arm",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_head_sampled_3.pkl --seq_len 50 --embedd_dim 72 --batch_size 32 --exp_name abl_param_pose_pelvis_arm_head",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_thorax_sampled_3.pkl --seq_len 50 --embedd_dim 117 --batch_size 32 --exp_name abl_param_pose_pelvis_arm_head_thorax",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_spine_sampled_3.pkl --seq_len 50 --embedd_dim 144 --batch_size 32 --exp_name abl_param_pose_pelvis_arm_head_thorax_spine",
    "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pose_leg_sampled_3.pkl --seq_len 50 --embedd_dim 198 --batch_size 32 --exp_name abl_param_pose_pelvis_arm_head_thorax_spine_leg"
]

joint_indiv_commands_v2 = [
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 15 --exp_name abl_ijoints_v3_pelvis_wrists",
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_elbows_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 21 --exp_name abl_ijoints_v3_pelvis_wrists_elbows",
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_elbows_eyes_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 27 --exp_name abl_ijoints_v3_pelvis_wrists_elbows_eyes",
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_elbows_eyes_head_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 33 --exp_name abl_ijoints_v3_pelvis_wrists_elbows_eyes_head",
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_elbows_eyes_head_ear_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 39 --exp_name abl_ijoints_v3_pelvis_wrists_elbows_eyes_head_ear",
]

sample_commands_v2 = [
   "cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_pelvis_wrists_elbows_sampled_50.pkl --batch_size 32 --seq_len 50 --embedd_dim 21 --exp_name abl_ijoints_v4_pelvis_wrists_elbows"
]

def gen_wrapper():
    run_commands(sample_commands_v2, exp_suffix="_f1_indices_fixed")

def run_commands(commands, exp_suffix=""):
    python_call = "/pasteur/u/bencliu/miniconda3/envs/or-hmr-pre2/bin/python"
    for command in commands:
        refactored_command = command.split(" ") 
        refactored_command[0] = python_call
        refactored_command[-1] = refactored_command[-1] + exp_suffix 
        subprocess.run(refactored_command)
        print("Finished running command") 

if __name__ == "__main__":
    print("Starting subprocess commands") 
    #testing() 
    gen_wrapper() 