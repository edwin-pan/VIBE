#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:22:43 2021

@author: edwinpan
"""


import numpy as np
import os
import joblib
import glob
import math

import metrics.utils as metric_utils
import metrics.moviemaker as mm
import metrics.frailty as frailty

generate_movie_flag = False

if __name__ == "__main__":
    # Load data
    VIBE_input_dir = "/Users/edwinpan/research/gait_motion/VIBE/data/gait_data_2"
    VIBE_output_dir = "/Users/edwinpan/research/gait_motion/VIBE/output/gait_data_2"
    video_name="default"
     # Find all files
    all_files_to_be_run = glob.glob(VIBE_output_dir+'/*/')
    for filenum, joints_dir in enumerate(all_files_to_be_run[14:15]):
        print(f"[{filenum}] Running: ", joints_dir)
        vibe_output = joblib.load(os.path.join(joints_dir, "vibe_output.pkl"))
    
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
        
        joints = joints3d
        joints = np.array(joints) * 1000 # some scaling is required, the auto-scaling doesn't quite work
        n = len(joints)
        for i in range(len(joints)):
            step = joints[i]
            for j in range(len(step)):
                joint = step[j]
                joints[i][j] = joint - step[0]

        print(joints.shape)
        metric_utils.summary_figure(joints, joints_dir, start=450, length=100, participantID=os.path.basename(os.path.normpath(joints_dir))[:3])
        
        print("summary PNG complete")

        if generate_movie_flag:
            # Modify the viewing angle
            # tx = math.radians(-90)
            # ty = math.radians(0)
            # tz = math.radians(-90) # used to be 160
            # rotx = [[1, 0, 0], [0, np.cos(tx), np.sin(tx)], [0, -np.sin(tx), np.cos(tx)]]
            # roty = [[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]]
            # rotz = [[np.cos(tz), np.sin(tz), 0], [-np.sin(tz), np.cos(tz), 0], [0, 0, 1]]
            # joints = joints.dot(rotx).dot(roty).dot(rotz)
            h36m_joints = [8, 27, 26, 22, 28, 29, 19, 41, 40, 37, 43, 5, 6, 7, 2, 3, 4] # h36m_joints is a subset of the VIBE/SPIN joints
            h36m_joint_set = joints[:, h36m_joints, :]
            
            walking_time, walking_speed, torso_angle, double_support_period, video_text, num_steps = frailty.angleFrailtyMetrics(h36m_joint_set, verbose_flag=True)
            
            metric_utils.generatePNG(joints, joints_dir)
            mm.movieMaker(os.path.join(joints_dir, 'animations/'), frame_rate=12, text=video_text, scanID="Test0-1-"+video_name, video_fname='L-ANGLE_0.avi', save_as_mp4_flag=True)
        
