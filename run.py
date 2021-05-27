# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:43:21 2020

@author: sumesh.thakur
"""

from tacos_dataset import Tacos_Caption,Vocabs,replace_map
from extractRGBfeatures import preprocess_image
from cnn_model import CNN
from rnn_model import CaptionGenerator
from tacos_dataset import DataLoader
import numpy as np
import argparse
import cv2
import pysrt
from tensorflow.keras.models import Sequential, save_model, load_model
import os
import pandas as pd
SAVEPATH = 'C:\\Users\\sumesh.thakur\\Downloads\\para\\feat'
CSV_PATH='.\\tacos_process.csv'
csv_file = pd.read_csv(CSV_PATH)

dataloader=DataLoader(CSV_PATH,SAVEPATH)

def main(arg):
    # build vocab
    CSV_PATH = '.\\tacos_process.csv'
    data=Tacos_Caption(CSV_PATH)
    captions = data['sentence'].values
    del data
    vocabs=Vocabs(list(replace_map(captions)))
    # build model
    cnn=CNN(arg.net) if arg.net else CNN()
    cnn.load_weights(arg.cnn_weight_path)
    rnn=CaptionGenerator(n_words=vocabs.n_words,
                         batch_size=1,
                         dim_feature=512,
                         dim_hidden=500,
                         n_video_lstm=64,
                         n_caption_lstm=20,
                         bias_init_vector=vocabs.bias_init_vector) 
    
    video_features,captions=dataloader.get_batch(batch_size=1)
    generators=rnn(video_features)
    
    rnn.load_weights(arg.rnn_weight_path,by_name=True)
    # extract video features
    print('extract %s video features' % (arg.video_path))
    this_features = []
    if arg.video_path.endswith('.avi'):
        cap = cv2.VideoCapture(arg.video_path)
        flame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  
        if flame_count > rnn.n_video_lstm:  
            select_flames = np.linspace(0, flame_count, num=rnn.n_video_lstm, dtype=np.int32)
        else:
            select_flames = np.arange(0, flame_count, dtype=np.int32)
        print('flame count:', flame_count, 'select: ', select_flames[:10])
        flames = []
        i, flame_index, selected_num = 0, 0, 0
        while True:
            ret, flame = cap.read()
            if ret is False:
                break
            if i == select_flames[flame_index]:
                flame_index += 1
                selected_num += 1
                flame = preprocess_image(flame)
                flames.append(flame)
            if selected_num == 64:
                selected_num = 0
                this_features.append(cnn.get_features(np.array(flames)))
                flames = []
            i += 1
            if i == flame_count and flames:
                this_features.append(cnn.get_features(np.array(flames)))
    else:
        raise ValueError("only support .avi video")
    #  generator caption
    this_features=np.concatenate(this_features,axis=0)
    this_feature_nums, dims_feature = this_features.shape
    if this_feature_nums < rnn.n_video_lstm: 
        this_features = np.vstack([this_features,
                                   np.zeros(shape=(rnn.n_video_lstm - this_feature_nums, dims_feature))])
    if this_feature_nums > rnn.n_video_lstm:  
        selected_idxs = np.linspace(0, this_feature_nums, num=rnn.n_video_lstm)
        this_features = this_features[selected_idxs, :]
    generator=rnn.predict(this_features.reshape(1,*this_features.shape))
    captions=[]
    for i, gen_caption in enumerate(generator):
        sent = []
        count = 0
        for ii in gen_caption:
        
            if ii == vocabs.word2idx['<eos>']:
                continue
            if ii == vocabs.word2idx['<pad>']:
                continue
            if ii != vocabs.word2idx['<bos>']:
                sent.append(vocabs.idx2word[ii])

        caption = ' '.join(sent)
        captions.append(caption)
    return captions

if __name__=="__main__":
    parse=argparse.ArgumentParser('generator')
    parse.add_argument('--net',type=str,default=None,help='vgg16 ,resnet50 or mobilenetv2(defalut)')
    parse.add_argument('--cnn_weight_path',type=str,default='.\\cnn_weights\\vgg16_notop.h5',help='the cnn model pretrained weight path')
    parse.add_argument('--rnn_weight_path',type=str,default='.\\save_model\\model_final.h5',help='the rnn model pretrained weight path')
    parse.add_argument('--video_path',type=str,default=None,help='the path of your video')

    arg=parse.parse_args()
    captions=main(arg)
    video_name = arg.video_path
    video_sub = os.path.basename(video_name[:-4])
    srt = pysrt.open("subtitle.srt")
    text_cap = []
    for cap in captions:
        print("caption: ",cap)
        text_cap.append(cap)
    sub = srt[0]
    sub.text = str(text_cap)
    sub.end.minutes = 0
    sub.end.seconds = 100
    sub.color = "#ffff00"
    srt.save(video_sub + ".srt", encoding='utf-8')