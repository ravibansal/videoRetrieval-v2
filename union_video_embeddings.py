# usage python union_video_embeddings.py <video_feature_folder> <video_feature_file.npy>

import numpy as np
import os

assert(len(sys.argv) == 3)

video_feat_path = sys.argv[1]

video_feats = sorted(os.listdir(video_path))

vid_feats = np.zeros([len(video_feats), 80, 4096])
i = 0
for video in video_feats:
	full_path = os.path.join(video_feat_path, video)
	vid_feats[i] = np.load(full_path)
	i = i + 1

print vid_feats.shape
np.save(sys.argv[2], vid_feats)
