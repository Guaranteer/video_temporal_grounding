import json
import pickle as pkl
import numpy as np
import random
import os
import nltk
import h5py
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader


def load_file(filename):
    with open(filename,'rb') as fr:
        return pkl.load(fr)

def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)

'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL


class Loader(Dataset):
    def __init__(self, params, key_file):
        #general
        self.params = params
        self.feature_path = params['feature_path']
        self.max_batch_size = params['batch_size']

        # dataset
        self.word2vec = KeyedVectors.load_word2vec_format(self.params["word2vec"], binary=True)
        # self.word_embedding = load_file(params['word_embedding'])
        # self.word2index = load_file(params['word2index'])
        self.key_file = load_json(key_file)
        self.dataset_size = len(self.key_file)


        # frame / question
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']



    def __getitem__(self, index):

        frame_vecs = np.zeros((self.max_frames, self.input_video_dim), dtype=float)

        ques_vecs = np.zeros((self.max_words, self.input_ques_dim), dtype=float)

        keys = self.key_file[index]
        vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]

        # video
        if not os.path.exists(self.feature_path + '/%s.h5' % vid):
            print('the video is not exist:', vid)
        with h5py.File(self.feature_path + '/%s.h5' % vid, 'r') as fr:
            feats = np.asarray(fr['feature'])
        real_n_frames = len(feats)
        n_frames = min(len(feats),self.max_frames)
        frame_vecs[:n_frames, :] = feats[:n_frames,:]
        frame_n = np.array(n_frames,dtype=int)

        # [64,128,256,512] / [16,32,64,128]
        frame_per_sec = real_n_frames/duration
        start_frame = round(frame_per_sec * timestamps[0])
        end_frame = round(frame_per_sec * timestamps[1]) - 1
        if start_frame > 767:
            start_frame = 703
            end_frame = 767
        if end_frame > 767:
            end_frame = 767
        widths = np.array([64, 128, 256, 512])
        nums = (768-widths*0.75)/(widths*0.25)
        nums = nums.astype(np.int) # [45,21,9,3]
        labels = [[0]*num for num in nums]
        cur_best = -2
        best_window = [0, 0]
        best_pos = [0, 63]
        for i in range(len(widths)):
            num = nums[i]
            width = widths[i]
            step = widths[i]/4
            for j in range(num):
                start = j*step
                end = start + width - 1

                iou = calculate_IoU([start_frame,end_frame],[start,end])
                niol = calculate_nIoL([start_frame,end_frame],[start,end])
                # if iou >= 0.5 and niol <= 0.2:
                if iou - niol >= cur_best:
                    cur_best = iou - niol
                    best_window = [i,j]
                    best_pos = [start, end]

        labels[best_window[0]][best_window[1]] = 1
        labels = np.hstack([labels[0],labels[1],labels[2],labels[3]])
        reg = np.array([start_frame - best_pos[0], end_frame - best_pos[1]])



        print(start_frame)
        print(end_frame)
        print(best_window)

        # question
        stopwords = ['.', '?', ',', '']
        sent = nltk.word_tokenize(sent)
        ques = [word.lower() for word in sent if word not in stopwords]
        # print(ques)
        ques = [self.word2vec[word] for word in ques if word in self.word2vec]
        ques_feats = np.stack(ques, axis=0)
        # print(len(ques))
        ques_n = np.array(min(len(ques), self.max_words),dtype=int)
        ques_vecs[:ques_n, :] = ques_feats[:ques_n, :]


        return frame_vecs, frame_n, ques_vecs, ques_n, labels, reg



    def __len__(self):

        return self.dataset_size


if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    train_dataset = Loader(config, config['train_data'])

    # Fetch one data pair (read data from disk).
    # frame_vecs, frame_n, ques_vecs, ques_n, labels, reg = train_dataset[0]
    # print(frame_vecs)
    # print(frame_vecs.shape)
    # print(frame_n)
    # print(ques_vecs)
    # print(ques_vecs.shape)
    # print(ques_n)
    # print(labels)
    # print(len(labels))

    # Data loader (this provides queues and threads in a very simple way).
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.

    frame_vecs, frame_n, ques_vecs, ques_n, labels, reg = data_iter.next()
    print(frame_vecs)
    print(frame_vecs.shape)
    print(frame_n)
    print(ques_vecs)
    print(ques_vecs.shape)
    print(ques_n)
    print(labels)
    print(len(labels))
    print(reg)


    # # Actual usage of the data loader is as below.
    # for frame_vecs, frame_n, ques_vecs, ques_n, labels, reg in train_loader:
    #     print(frame_vecs)
    #     print(frame_vecs.shape)
    #     print(frame_n)
    #     print(ques_vecs)
    #     print(ques_vecs.shape)
    #     print(ques_n)
    #     print(labels)
    #     print(len(labels))
