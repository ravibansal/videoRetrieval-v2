from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import pandas as pd

class DatasetLoader:
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def __init__(self, vd_feat_path, sent_feat_path, split='train'):
        print('Loading video features from', vd_feat_path)
        
        ##Data reading from video data path, old code
        vd_feats = np.load(vd_feat_path)
        
        # im_feats = np.array(data_im['image_features']).astype(np.float32)
        print('Loaded video feature shape:', vd_feats.shape)
        print('Loading sentence features from', sent_feat_path)
        sent_feats = np.load(sent_feat_path)
        print('Loaded sentence feature shape:', sent_feats.shape)

        self.split = split
        self.vd_feat_shape = vd_feats.shape
        self.sent_feat_shape = sent_feats.shape
        self.sent_inds = range(len(sent_feats)) # we will shuffle this every epoch for training
        self.vd_feats = vd_feats
        self.sent_feats = sent_feats
        # Assume the number of sentence per video is a constant.
        self.sent_im_ratio = len(sent_feats) // len(vd_feats)

    def shuffle_inds(self):
        '''
        shuffle the indices in training (run this once per epoch)
        nop for testing and validation
        '''
        if self.split == 'train':
            np.random.shuffle(self.sent_inds)
            #np.random.shuffle(self.im_inds)

    def sample_items(self, sample_inds, sample_size):
        '''
        for each index, return the  relevant video and sentence features
        sample_inds: a list of sent indices
        sample_size: number of neighbor sentences to sample per index.
        '''
        vd_feats_b = self.vd_feats[[i // self.sent_im_ratio for i in sample_inds],:]
        sent_feats_b = []
        for ind in sample_inds:
            # ind is an index for sentence
            start_ind = ind - ind % self.sent_im_ratio
            end_ind = start_ind + self.sent_im_ratio
            sample_index = np.random.choice(
                    [i for i in range(start_ind, end_ind) if i != ind],
                    sample_size - 1, replace=False)
            sample_index = sorted(np.append(sample_index, ind))
            sent_feats_b.append(self.sent_feats[sample_index])
        sent_feats_b = np.concatenate(sent_feats_b, axis=0)
        return (vd_feats_b, sent_feats_b)

    def get_batch(self, batch_index, batch_size, sample_size):
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        if self.split == 'train':
            sample_inds = self.sent_inds[start_ind : end_ind]
        else:
            # Since sent_inds are not shuffled, every self.sent_im_ratio sents
            # belong to one video. Sample each pair only once.
            sample_inds = self.sent_inds[start_ind * self.sent_im_ratio : \
                            end_ind * self.sent_im_ratio : self.sent_im_ratio]
        (vd_feats, sent_feats) = self.sample_items(sample_inds, sample_size)
        # Each row of the labels is the label for one sentence,
        # with corresponding video index sent to True.
        labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return(vd_feats, sent_feats, labels)
