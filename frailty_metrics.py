#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:40:47 2020

@author: edwinpan
"""


import numpy as np
import os
import math
import joblib
import glob

import imageio
import lib.data_utils.kp_utils as utils

import metrics.utils as metric_utils
import metrics.moviemaker as moviemaker
import metrics.frailty as frailty

if __name__ == "__main__":
    # Experiment Hyperparameters
    knee_angle_tolerance = 155 # Degrees
    past_frame_memory_delta = 2 # Frames

    hyperparam_dict = {"Knee Angle Tolerance (degrees)":knee_angle_tolerance, 
                       "Temporal Comparison Offset (frames)":past_frame_memory_delta}
    
    # Initialize Experiment Recorder
    columns = ["Filename", "Walking Time", "Walking Speed", "Average Torso Angle", "Double Support Period", "[OPTIONAL] Number of Steps"]
    record = metric_utils.experimentRecorder("FrailtyMetricExperiment_scratch.xlsx", columns, hyperparam_dict)
    
    # Load data
    VIBE_input_dir = "/Users/edwinpan/research/gait_motion/VIBE/data/gait_data_2"
    VIBE_output_dir = "/Users/edwinpan/research/gait_motion/VIBE/output/gait_data_2"
    
    # Find all files
    all_files_to_be_run = glob.glob(VIBE_output_dir+'/*/')
    for filenum, joints_dir in enumerate(all_files_to_be_run):
        print(f"[{filenum}] Running: ", joints_dir)
        video_name = os.path.basename(os.path.normpath(joints_dir))
        vibe_output = joblib.load(os.path.join(joints_dir, "vibe_output.pkl"))
        
        if video_name.lower().endswith(('.MOV', '.mov')):
            input_video_fname = os.path.join(VIBE_input_dir, video_name)
        else:
            input_video_fname = os.path.join(VIBE_input_dir, video_name+'.MOV')
    
        original_video = imageio.get_reader(input_video_fname,  'ffmpeg')
        metadata = original_video.get_meta_data()
    
        print("[NOTE] Number of people detected: ", len(vibe_output.keys()))  # Number of unique people tracked in the video
        
        tracks = vibe_output.keys()
        
        bboxes = vibe_output[1]["bboxes"]
        betas = vibe_output[1]["betas"]
        frame_ids = vibe_output[1]["frame_ids"]
        joints2d = vibe_output[1]["joints2d"]
        joints3d = vibe_output[1]["joints3d"]
        orig_cam = vibe_output[1]["orig_cam"]
        pose = vibe_output[1]["pose"]
        pred_cam = vibe_output[1]["pred_cam"]
        verts = vibe_output[1]["verts"]
        
        # Constants & Details
        deepblue = "#00107A"
        mediumblue = "#1668EB"
        mediumred = "#e74c3c"
        # joints_dir = '/Users/tacobell/research/svl/EPD_data/J/'
        # joints_dir = '/users/edwinpan/research/gait_motion/VIBE/'
        moviemaker_test_dir = '/users/edwinpan/research/gait_motion/VIBE/tests/test_moviemaker/seal'
        clip = 'default'
        
            
        joints = joints3d
        joints = np.array(joints) * 1000 # some scaling is required, the auto-scaling doesn't quite work
        n = len(joints)
        for i in range(len(joints)):
            step = joints[i]
            for j in range(len(step)):
                joint = step[j]
                joints[i][j] = joint - step[0]
        
        # animation_dir = generatePNG(joints, joints_dir) # Generate PNGs for videos
        
        # Modify the viewing angle
        tx = math.radians(-90)
        ty = math.radians(0)
        tz = math.radians(-90) # used to be 160
        rotx = [[1, 0, 0], [0, np.cos(tx), np.sin(tx)], [0, -np.sin(tx), np.cos(tx)]]
        roty = [[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]]
        rotz = [[np.cos(tz), np.sin(tz), 0], [-np.sin(tz), np.cos(tz), 0], [0, 0, 1]]
        joints = joints.dot(rotx).dot(roty).dot(rotz)
        h36m_joints = [8, 27, 26, 22, 28, 29, 19, 41, 40, 37, 43, 5, 6, 7, 2, 3, 4] # h36m_joints is a subset of the VIBE/SPIN joints
        h36m_joint_set = joints[:, h36m_joints, :]
            
            
        # print("Number of frames: ", joints.shape[0])
        # print("Number of h36m joints: ", len(h36m_joints))
        # print("keys: ", utils.get_h36m_joint_names())
        # print(h36m_joint_set.shape)
        
        # Define dict mapping between names and joint indices in 
        h36m_joint_dict = {joint_name : joint_idx for joint_name, joint_idx in zip(utils.get_h36m_joint_names(), h36m_joints)}
        # print(h36m_joint_dict)
        
        # movieMaker(os.path.join(joints_dir, 'animations/'), frame_rate=10, scanID="Test0-0-"+video_name+"-axes", video_fname='LANKLE_distances.avi')
        
        walking_time, walking_speed, torso_angle, double_support_period, video_text, num_steps = frailty.angleFrailtyMetrics(h36m_joint_set, metadata=metadata, verbose_flag=True, knee_angle_tol=knee_angle_tolerance, delta=past_frame_memory_delta)
        
        print()
        print("--- FINAL METRICS ---")
        print("Walking Time: \t\t", walking_time, " sec")
        print("Walking Speed: \t\t", walking_speed, " 'world-units'/sec")
        print("Average Torso angle: \t", torso_angle, " degrees")
        print("Double Support Period: \t", double_support_period, " sec")
        print("---------------------")
        print()
    
        record.add_row([video_name, walking_time, walking_speed, torso_angle, double_support_period, num_steps])
        
        # movieMaker(os.path.join(joints_dir, 'animations/'), frame_rate=12, text=video_text, scanID="Test0-1-"+video_name, video_fname='L-ANGLE_0.avi', save_as_mp4_flag=True)

    record.close()