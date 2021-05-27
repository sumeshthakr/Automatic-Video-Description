# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:18:46 2020

@author: sumesh.thakur
"""

import pandas as pd
import numpy as np
import cv2
import os

tacos_data = pd.read_csv("./tacos_server.csv")

def process_video(path,video_id,start,end):
    start_frame = int(start)
    end_frame = int(end)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(video_id+"_"+str(start_frame)+"_"+str(end_frame)+".avi", fourcc, fps, (w,h))
    
    frame_count = 0
    while frame_count < end_frame:
        ret, frame = cap.read()
        frame_count += 1
        if frame_count >= start_frame:
            out.write(frame)
    cap.release()
    out.release()
    
def get_name(x):
    all_enty = []
    for idx in range(len(x)):
        video_id = x.iloc[idx,3]
        path=x.iloc[idx,6]
        caption=x.iloc[idx,2]
        start = x.iloc[idx,4]
        end = x.iloc[idx,5]
        all_enty.extend([[video_id,path,caption,start,end]])
    return all_enty

all_enteries = get_name(tacos_data)    

for i in range(len(all_enteries)):
    process_video(all_enteries[i][1],all_enteries[i][0],all_enteries[i][3],all_enteries[i][4])   