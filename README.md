# videoRetrieval-v2
## Instruction for the data to be used:
1. Please make sure that the video-sentence ratio is constant.
2. Organize your video data such that, you have all the video files for training inside "data_train" folder and for testing inside "data_test" folder. (You have the option to provide your own file names in the input/argumemts to the python files).
3. Organize your sentence data such that, a "setences_train.txt" OR 'sentences_test.txt" file contains all the sentences corresponding to train/test data. The order of the sentences should be same as the order of the videos (which is sorted alphabetically). So, for example if you have two videos "a.mp4" and "b.mp4" in your video directory and the dataset contains 10 sentences/annotations per video then the sentences.txt file should contain 10 sentences for "a.mp4" than 10 sentences for "b.mp4" with each sentence in a new line in the text file.
4. Make different .npy files for train and test data using the below preprocessing files. The resulting feature files will be:
    1. vid_feats_train.npy
    2. sent_feats_train.npy
    3. vid_feats_test.npy
    4. sent_feats_test.npy
## extract_video_embeddings.py : 
To extract video feature of size [80 4096] for all the videos in a given path and save it into a provided path. (using pre-trained VGG16 model). Download pretrianed npy file from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
## extract_sentence_embeddings.py :
To extract sentence feature of size [4800] for all the sentences in a given .txt file and save it into a provided path (as .npy file).
### -- Requirements: <br />
Please follow https://github.com/ryankiros/skip-thoughts to dowload pre-trained skip-thought model. Please keep extract_sentence_embeddings.py python file in the same directory as skipthoughts.py
<br />
### --usage: 
python extract_sentence_embeddings.py <sentence_input_file> <sentence_feature_file.npy>
<br />
## union_video_embeddings.py :
To take union of all the video features generated by extract_video_embeddings.py and create one .npy file of size [num of videos, 80, 4096].
<br />
### --usage:
python union_video_embeddings.py <video_feature_folder> <video_feature_file.npy>
## train_embedding_nn.py :
To train the two branch network on the train data with the input as video feature and sentence feature path. It will checkpoint the model for last n timestamps in provided checkpoint directory.
### --usage:
vid_feats.npy: [num_of_videos, 80, 4096]
<br />
sent_feats.npy: [num_of_sentences, 4800]
<br />
python train_embedding_nn.py \
    --video_feat_path ./vid_feats.npy \
    --sent_feat_path ./sent_feats.npy \
    --save_dir ./checkpoint/
## eval_embedding_nn.py :
To evaluate the model for test data with the input as video feature and sentence feature path. It will restore the model from checkpoint directory and use it produce recall@K(1, 5, 10) values for vid2sent and sent2vid.
### --usage:
vid_feats.npy: [num_of_videos, 80, 4096]
<br />
sent_feats.npy: [num_of_sentences, 4800]
<br />
python eval_embedding_nn.py \
    --video_feat_path ./vid_feats_test.npy  \
    --sent_feat_path ./sent_feats_test.npy \
    --restore_path ./checkpoint/-4081.meta \
    --sample_size <Setences-per-video>
