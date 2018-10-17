import sys
sys.path.append('..')
import json
import pickle as pkl
import os
import numpy as np
import h5py
import nltk

def load_file(filename):
    with open(filename,'rb') as fr:
        return pkl.load(fr)

def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)


def dataset_build(config):

    raw_files = [config['train_json'], config['val_json'], config['test_json']]
    processed_files = [config['train_data'],config['val_data'],config['test_data']]

    for i in range(3):
        raw_file = raw_files[i]
        processed_file = processed_files[i]

        processed_data = list()

        with open(raw_file, 'r') as fr:
            raw_data = json.load(fr)
        for vid, data in raw_data.items():
            duration = data['duration']
            timestamps = data['timestamps']
            sentences = data['sentences']
            for idx in range(len(timestamps)):
                processed_data.append([vid,duration,timestamps[idx],sentences[idx]])
        print('sample num:', len(processed_data))
        json.dump(processed_data,open(processed_file,'w'))


def longest_video(config):
    data_files = [config['train_data'], config['val_data'], config['test_data']]
    long = 0
    length_set = dict()
    for data_file in data_files:
        key_file = load_json(data_file)
        for keys in key_file:
            vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]

            # stopwords = ['.', '?', ',', '']
            # sent = nltk.word_tokenize(sent)
            # ques = [word.lower() for word in sent if word not in stopwords]
            # if len(ques) > long:
            #     long = len(ques)

            # video
            if config['is_origin_dataset']:
                if not os.path.exists(config['feature_path'] + '/%s.h5' % vid):
                    print('the video is not exist:', vid)
                with h5py.File(config['feature_path'] + '/%s.h5' % vid, 'r') as fr:
                    feats = np.asarray(fr['feature'])
                    if len(feats) > long:
                        long = len(feats)
                    if len(feats) in length_set:
                        length_set[len(feats)] += 1
                    else:
                        length_set[len(feats)] = 1


    print(long)
    print(length_set)

if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    # dataset_build(config)
    longest_video(config)
