# videoRetrieval-v2
Files and its usage
# extract_video_embeddings.py : 
To extract video feature of size [80 4096] for all the videos in a given path and save it into a provided path. (using pre-trained VGG16 model). Download pretrianed npy file from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
# extract_sentence_embeddings.py :
-- Requirements: <br />
Please follow https://github.com/ryankiros/skip-thoughts to dowload pre-trained skip-thought model. Please keep this python file in the same directory as skipthoughts.py
<br />
--usage: python extract_sentence_embeddings.py <sentence_input_file> <sentence_feature_file.npy>
<br />
To extract sentence feature of size [4800] for all the sentences in a given .txt file and save it into a provided path (as .npy file).
