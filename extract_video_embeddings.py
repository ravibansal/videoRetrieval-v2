#-*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import skimage
import tensorflow as tf
import vgg16


def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def main():
    num_frames = 80
    video_path = './data'
    video_save_path = './data_features'
    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)
    videos = os.listdir(video_path)
    # videos = filter(lambda x: x.endswith('avi'), videos)

    # Load the CNN model trained on images to recognize objects.
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            vgg = vgg16.Vgg16()
            for idx, video in enumerate(videos):
                print idx, video

                if os.path.exists( os.path.join(video_save_path, video) ):
                    print "Already processed ... "
                    continue

                video_fullpath = os.path.join(video_path, video)
                try:
                    cap  = cv2.VideoCapture( video_fullpath )
                except:
                    print "cv2 not working"
                    pass

                frame_count = 0
                frame_list = []

                while True:
                    ret, frame = cap.read()
                    if ret is False:
                        break

                    frame_list.append(frame)
                    frame_count += 1

                frame_list = np.array(frame_list)
                if frame_count > 80:
                    frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
                    frame_list = frame_list[frame_indices]

                processed_frame_list = []
                for x in frame_list:
                    processed_frame_list.append(preprocess_frame(x))

                # Generate the frame list of equally spaced 80 frames.
                cropped_frame_list = np.array(processed_frame_list)
                cropped_frame = np.zeros([80, 224, 224, 3])
                cropped_frame[:cropped_frame_list.shape[0]] = cropped_frame_list
                print "Cropped Frame List: ", cropped_frame.shape
                images = tf.placeholder("float", [cropped_frame.shape[0], 224, 224, 3])
                feed_dict = {images: cropped_frame}
                with tf.name_scope("content_vgg"):
                    vgg.build(images)
                feats = sess.run(vgg.fc6, feed_dict=feed_dict)
                print "Feature Length: ", len(feats)
                save_full_path = os.path.join(video_save_path, video + '.npy')
                np.save(save_full_path, feats)

if __name__=="__main__":
    main()
    
