# usage python extract_sentence_embeddings.py <sentence_input_file> <sentence_feature_file.npy>

import pickle
import sys
import skipthoughts
import numpy as np


assert(len(sys.argv) == 3)

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

sentences = open(sys.argv[1])

sent_feats = []

i = 0
for sent in sentences:
	sent = sent.strip()
	query = sent.lower()
	vec = encoder.encode([query])[0]
	sent_feats.append(vec)
	i = i + 1
	if i%10 == 0:
		print "completed_sentences: ", i

sent_feats = np.array(sent_feats)
print sent_feats.shape
np.save(sys.argv[2], sent_feats)