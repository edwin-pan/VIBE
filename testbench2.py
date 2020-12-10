#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:56:03 2020

@author: edwinpan
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
# import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math
import joblib

import cv2
import imageio

# New imports
import lib.data_utils.kp_utils as utils


"""Functions to visualize human poses"""
class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", linewidth=2, linestyle='-', alpha=1):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
        J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
        
        # joint_map which uses OP joints for everything except lower body
        joint_map = {0:8, 1:27, 2:26, 3:22, 6:28, 7:29, 8:19, 12:41, 13:40, 14:37, 15:43, 17:5, 18:6, 19:7, 25:2, 26:3, 27:4}
        
        self.I = np.ones(16, dtype=np.int32)
        self.J = np.ones(16, dtype=np.int32)
        
        for i in range(len(I)):
            self.I[i] = joint_map[I[i]]
            self.J[i] = joint_map[J[i]]
            
        # Left / right indicator
        self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((49, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange( len(self.I) ):
            x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
            y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
            z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
            self.plots.append(self.ax.plot(x, y, z, 
                                           lw=linewidth, 
                                           c=lcolor if self.LR[i] else rcolor,
                                           linestyle=linestyle,
                                           alpha=alpha))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        vals = np.reshape(channels, (49, -1))

        for i in np.arange( len(self.I) ):
            x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
            y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
            z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

        r = 750;
        xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
        self.ax.set_xlim3d([-r+xroot, r+xroot])
        self.ax.set_zlim3d([-r+zroot, r+zroot])
        self.ax.set_ylim3d([-r+yroot, r+yroot])
        

"""Function to Generate PNGs in specified directory"""
def generatePNG(joints, joints_dir):
    #  ------ GENERATE PNG ------
    # === Plot and animate ===
    fig = plt.figure(dpi=200)
    
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
    
    # Define the range and frames to display
    start = 0
    end = joints.shape[0]
    
    # Modify axes and background 
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot
    animation_dir = os.path.join(joints_dir, 'animations/')
    if not os.path.isdir(animation_dir):
        os.mkdir(animation_dir)
    else:
        print(animation_dir, " already exists")
    
    
    print("[INFO] Generating PNGs in ", animation_dir)
    for i in range(start, end):
        # modify colors, line-width, line-style and fading to create custom effect
        ax.clear()
        ax = plt.gca(projection='3d')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.view_init(elev=10, azim=90)
        plt.axis('off')
        plt.grid(False)
    
        ob = Ax3DPose(ax, linewidth=1.3, linestyle='-')
        ob.update(joints[i,:, :], lcolor=mediumblue, rcolor=mediumred)
        ax.scatter(h36m_joint_set[i, :, 0], h36m_joint_set[i, :, 1], h36m_joint_set[i, :, 2], s=6, c=deepblue, alpha=1)
        plt.savefig(os.path.join(animation_dir,clip + '_gait0_reg_'+str(i)+'.png')) # save image
        
    # Draw darkest frame
    ob = Ax3DPose(ax, linewidth=1, linestyle='-')
    ob.update(joints[start,:, :], lcolor=deepblue, rcolor=deepblue)
    plt.show(block=False)
    fig.canvas.draw()
    #  ------ GENERATE PNG ------
    return animation_dir


# ----- VIDEO MAKING -----
# Save video
import glob
import os
import cv2

RUN_TESTS_FLAG=False

def movieMaker(png_dir, text=None, save_as_mp4_flag=False, frame_rate = 15, clip='default', video_fname='gait0_full.avi', verbose_flag=False, scanID='default'):
    """ Create a video file from all PNG files in a specified directory
    
        text should be a list of strings
    """
    
    if not os.path.isdir(png_dir):
        raise ValueError("movieMaker was given a non-existent directory")
    else:
        print("[movieMaker] Grabbing images from: ", png_dir)
    image_filepath_list = sorted(glob.glob(os.path.join(png_dir, '*.png')), key=os.path.getmtime)

    if not len(image_filepath_list):
        raise ValueError("movieMaker was given a directory with no .png files")
        
    video_name = os.path.join(png_dir, video_fname)
    print("[movieMaker] Saving output as: ", video_name)

    # Create Video
    frame = cv2.imread(image_filepath_list[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, frame_rate, (width,height))

    # Font Utilities
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,height-10)
    topLeftCorderOfText    = (10,20)
    fontScale              = 0.75
    fontColor              = (21, 21, 140) # NOTE: OpenCV is BGR
    lineType               = 2
    
    for idx, image_filepath in enumerate(image_filepath_list):
        img = cv2.imread(image_filepath)
        
        cv2.putText(img,scanID, 
            topLeftCorderOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
                    
        if text is not None:
            cv2.putText(img,text[idx], 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
            
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    
    
    if save_as_mp4_flag:
        import subprocess
        output_name = video_name[:-4]+'.mp4'
        subprocess.call(['ffmpeg', '-i', video_name, output_name])
        os.remove(video_name)
        
    return

# if RUN_TESTS_FLAG:
#     movieMaker(moviemaker_test_dir, frame_rate=5, video_fname='seal.avi')
# ----- VIDEO MAKING -----
        

"""Compute Euclidean Frailty Metrics"""
def euclideanFrailtyMetrics(verbose_flag=False):
    step_condition_log = []
    # -----------------------------
    # D (T) TUG Test Total Time
    # D (\tau) Walking Time
    # D (\delta) Covered Distance
    # D (\beta) Walking Speed
    # D (\zeta) Swing Time
    # D (\eta) Double Support Time
    # D (\phi) Torso Incline Angle
    # -----------------------------
    
    # Note: Distance metric = Euclidean (L2)
    # Hyper-parameters
    d_tol = 100 #cm TODO: new stride detection threshold
    delta = 2
    
    # Flags
    is_left_foot_moving_flag = False
    towards_camera_flag = True
    VERBOSE_DEBUG_flag = False
    
    metric_delta = 0
    metric_zeta = 0
    ts_log = []
    te_log = []
    phi_log = []
    
    start = 0
    for frame_idx, frame in enumerate(h36m_joint_set[start:,:,:]):
        if VERBOSE_DEBUG_flag:
            print("Iteration: ", frame_idx+start)
        ANKLE_LEFT_coord_at_current_idx = frame[3,:] # Index 3 = left-ankle
        
        if frame_idx < delta:
            ANKLE_LEFT_coord_at_prev_idx = h36m_joint_set[start,3,:] # Index 3 = left-ankle
        else:
            ANKLE_LEFT_coord_at_prev_idx = h36m_joint_set[frame_idx-delta,3,:] # Index 3 = left-ankle
            
            # Check direction of walk -- TODO: Needs Fixing
            if frame[0,0] - h36m_joint_set[frame_idx-delta,0,0] > 0:
                towards_camera_flag = False
            
        # (\phi) Torso Incline Angle
        c = np.array([1,0]) # Index 0 = Hip
        s = frame[8,:]-frame[7,:] # Hip (0) -> Spine (7) -> Neck (8)
        s = np.array([s[0], s[2]]) # Remove y axis
        curr_phi = np.arccos(np.dot(c, s/np.linalg.norm(s)))
        
        # print(c,s/np.linalg.norm(s), round(min(curr_phi*180/(np.pi), 180-curr_phi*180/(np.pi)), 2))
        curr_phi_degree = min(curr_phi*180/(np.pi), 180-curr_phi*180/(np.pi))
        
        ankle_distance = np.linalg.norm(ANKLE_LEFT_coord_at_current_idx - ANKLE_LEFT_coord_at_prev_idx)
        step_condition_log.append('Frame: '+ str(frame_idx) + ' Phi: ' + str(round(curr_phi_degree, 2)) + ' L-ANKLE Distance: ' + str(round(ankle_distance, 4)))
        phi_log.append(curr_phi_degree)
        
        if ankle_distance > d_tol:
            if VERBOSE_DEBUG_flag:
                print("[NOTE] Ankle has risen with dist: ", ankle_distance)
            if not is_left_foot_moving_flag:
                is_left_foot_moving_flag = True
                ts = frame_idx # TODO: Time is computed in frames. Convert to seconds using frame-rate
                ts_log.append(ts)
                SPINE_BASE_ts = frame[7,:]# Motion has started. Record starting position
        else:
            if is_left_foot_moving_flag:
                if VERBOSE_DEBUG_flag:
                    print("[Note] Ankle has come down")
                is_left_foot_moving_flag = False
                te = frame_idx # TODO: Time is computed in frames. Convert to seconds using frame-rate
                te_log.append(te)
                
                # Update metrics
                SPINE_BASE_te = frame[7,:]
                metric_delta += np.linalg.norm(SPINE_BASE_ts-SPINE_BASE_te) 
                metric_zeta += te-ts
    
    # (T) TUG Test Total Time
    metric_T = h36m_joint_set.shape[0]/metadata['fps']
    
    # (\tau) Walking Time (NEED)
    metric_tau = (te_log[-1] - ts_log[0])/metadata['fps']
    
    # (\eta) Double Support Time (NEED)
    metric_eta = metric_tau - (metric_zeta/metadata['fps'])
    
    # (\beta) Walking Speed (NEED)
    metric_beta = metric_delta/metric_tau
    
    if verbose_flag:
        return metric_tau, metric_beta, np.average(phi_log), metric_eta, step_condition_log, phi_log
    else:
        return metric_tau, metric_beta, np.average(phi_log), metric_eta


"""Compute Euclidean Frailty Metrics"""
def angleFrailtyMetrics(h36m_joint_set, verbose_flag=False):
    tracker_text = []
    # -----------------------------
    # D (T) TUG Test Total Time
    # D (\tau) Walking Time
    # D (\delta) Covered Distance
    # D (\beta) Walking Speed
    # D (\zeta) Swing Time
    # D (\eta) Double Support Time
    # D (\phi) Torso Incline Angle
    # -----------------------------
    
    # Note: Distance metric = Euclidean (L2)
    # Hyper-parameters
    knee_angle_tol = 155 #degrees
    delta = 2
    
    # Flags
    is_left_foot_moving_flag = False
    towards_camera_flag = True
    VERBOSE_DEBUG_flag = False
    
    metric_delta = 0
    metric_zeta = 0
    ts_log = []
    te_log = []
    phi_log = []
    
    start = 0
    for frame_idx, frame in enumerate(h36m_joint_set[start:,:,:]):
        if VERBOSE_DEBUG_flag:
            print("Iteration: ", frame_idx+start)
            
            # Check direction of walk -- TODO: Needs Fixing
            if frame[0,0] - h36m_joint_set[frame_idx-delta,0,0] > 0:
                towards_camera_flag = False
            
        # (\phi) Torso Incline Angle
        c = np.array([1,0]) # Index 0 = Hip
        s = frame[0,:]-frame[7,:] # Hip (0) -> Spine (7) -> Neck (8)
        s = np.array([s[0], s[2]]) # Remove y axis
        curr_phi = np.arccos(np.dot(c, s/np.linalg.norm(s)))
        
        curr_phi_degree = min(curr_phi*180/(np.pi), 180-curr_phi*180/(np.pi))
        phi_log.append(curr_phi_degree)

        # Track walking state
        left_ankle_coord = np.array([frame[3,0], frame[3,2]])
        left_knee_coord = np.array([frame[2,0], frame[2,2]])
        left_hip_coord = np.array([frame[1,0], frame[1,2]])
        thigh_vector = (left_hip_coord-left_knee_coord)/np.linalg.norm(left_hip_coord-left_knee_coord)
        calf_vector = (left_ankle_coord-left_knee_coord)/np.linalg.norm(left_ankle_coord-left_knee_coord)

        left_knee_angle = np.arccos(np.dot(thigh_vector, calf_vector))*180/np.pi
        tracker_text.append('Frame: '+ str(frame_idx) + ' Phi: ' + str(round(curr_phi_degree, 2)) + ' L-Knee-Angle: ' + str(round(left_knee_angle, 4)))        
        if left_knee_angle > knee_angle_tol:
            if VERBOSE_DEBUG_flag:
                print("[NOTE] Left-knee at angle: ", left_knee_angle)
            if not is_left_foot_moving_flag:
                is_left_foot_moving_flag = True
                ts = frame_idx # TODO: Time is computed in frames. Convert to seconds using frame-rate
                ts_log.append(ts)
                SPINE_BASE_ts = frame[7,:]# Motion has started. Record starting position
        else:
            if is_left_foot_moving_flag:
                if VERBOSE_DEBUG_flag:
                    print("[Note] Ankle has come down")
                is_left_foot_moving_flag = False
                te = frame_idx # TODO: Time is computed in frames. Convert to seconds using frame-rate
                te_log.append(te)
                
                # Update metrics
                SPINE_BASE_te = frame[7,:]
                metric_delta += np.linalg.norm(SPINE_BASE_ts-SPINE_BASE_te) 
                metric_zeta += te-ts
    
    # (T) TUG Test Total Time
    metric_T = h36m_joint_set.shape[0]/metadata['fps']
    
    # (\tau) Walking Time (NEED)
    if len(te_log) and len(ts_log):
        print("[NOTE] ", len(te_log), " steps detected")
        metric_tau = (te_log[-1] - ts_log[0])/metadata['fps']
    else:
        print("[WARNING] No Steps Detected")
        metric_tau = 1
    
    # (\eta) Double Support Time (NEED)
    metric_eta = metric_tau - (metric_zeta/metadata['fps'])
    
    # (\beta) Walking Speed (NEED)
    metric_beta = metric_delta/metric_tau
    
    if verbose_flag:
        return metric_tau, metric_beta, np.average(phi_log), metric_eta, tracker_text, phi_log
    else:
        return metric_tau, metric_beta, np.average(phi_log), metric_eta
    
    
if __name__ == "__main__":
    # Load data
    video_name = "IMG_1397"
    joints_dir = "/Users/edwinpan/research/gait_motion/VIBE/output/VideoStudy1/" + video_name
    vibe_output = joblib.load(os.path.join(joints_dir, "vibe_output.pkl"))
    video_fname = os.path.join(joints_dir, video_name+".mov_vibe_result.mp4")
    
    # original_video = cv2.VideoCapture(video_fname)
    original_video = imageio.get_reader(video_fname,  'ffmpeg')
    metadata = original_video.get_meta_data()
    
    print(vibe_output.keys())  # Number of unique people tracked in the video
    print(vibe_output[1].keys()) # Data for each subject
    
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
    
    walking_time, walking_speed, torso_angle, double_support_period, video_text, _ = angleFrailtyMetrics(h36m_joint_set, verbose_flag=True)
    

    
    print()
    print("--- FINAL METRICS ---")
    print("Walking Time: \t\t", walking_time, " sec")
    print("Walking Speed: \t\t", walking_speed, " 'world-units'/sec")
    print("Average Torso angle: \t", torso_angle, " degrees")
    print("Double Support Period: \t", double_support_period, " sec")
    print("---------------------")
    print()

    # movieMaker(os.path.join(joints_dir, 'animations/'), frame_rate=12, text=video_text, scanID="Test0-1-"+video_name, video_fname='L-ANGLE_0.avi', save_as_mp4_flag=True)
