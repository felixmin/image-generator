import os
import cv2
import glob
import numpy as np
import tensorflow as tf

############## GOOD FAST VERSION ##############

def load_images(batch_size, shuffle_buffer_size=1024, path='/Users/felixminzenmay/Code/datasets/ffhq/imgs/*'):
    # Use glob to efficiently find all PNG files
    filenames = glob.glob(os.path.join(path, '*.png'))

        
    def load_img(filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # Use tf.data.Dataset.from_generator for parallel loading
    def generator():
        for filename in filenames:
            yield load_img(filename)


    dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8))
    
    def normalize_img(img):
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    dataset = dataset.map(normalize_img)

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    batched_dataset = dataset.batch(batch_size)
    return batched_dataset

############## BAD SLOW VERSION ##############

# def load_images(batch_size, path='/Users/felixminzenmay/Code/datasets/ffhq'):
#     filenames = []

#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".png"):
#                 filename = root + "/" + file
#                 filenames.append(filename)
            
#     def load_img(filename):
#         img = cv2.imread(filename)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
    
#     images = []
#     for filename in filenames:
#         images.append(load_img(filename))

#     imgs_arr = np.array(images)
#     dataset = tf.data.Dataset.from_tensor_slices(imgs_arr)
#     batched_dataset = dataset.batch(batch_size)
#     return batched_dataset


