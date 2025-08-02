import os
from utils.img_read_save import image_read_cv2
import h5py
import numpy as np
from tqdm import tqdm
import cv2
from utils.img_read_save import img_save
img_text_path = 'VLFDataset'
h5_path = "VLFDataset_h5"
os.makedirs(h5_path, exist_ok=True)
task_name = 'MFF'
dataset_name = 'Pathology'
dataset_mode = 'test'
h5_file_path = os.path.join(h5_path, dataset_name + '_' + dataset_mode + '.h5')
if dataset_name == 'RealMFF' or dataset_name == 'Lytro':
    small_size_0 = (384, 288)
    small_size_1 = (288, 384)
elif dataset_name == 'Pathology':
    small_size_0 = (256, 256)
    small_size_1 = (256, 256)
with h5py.File(h5_file_path, 'w') as h5_file:
    imageA = h5_file.create_group('imageA')
    imageB = h5_file.create_group('imageB')
    textA = h5_file.create_group('textA')
    textB = h5_file.create_group('textB')
    text_txt = os.path.join(img_text_path, 'Image', task_name, dataset_name, dataset_mode + '.txt')
    with open(text_txt, 'r') as file:
        file_list = [line.strip() for line in file.readlines()]
    sample_names = []
    for name in file_list:
        name = name.split('.')[0]
        sample_names.append(name)
    for sample_name in tqdm(sample_names):
        if dataset_mode == 'train':
            '''img_A = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'NEAR', sample_name + '.png'),mode='RGB')/ 255.0
            img_B = image_read_cv2(os.path.join(img_text_path, 'Image', task_name, dataset_name, 'FAR', sample_name + '.png'),mode='RGB')/ 255.0
            img_A = cv2.resize(img_A, img_size).transpose(2,0,1)
            img_B = cv2.resize(img_B, img_size).transpose(2,0,1)'''
            img_A = image_read_cv2(
                os.path.join(img_text_path, 'Image', task_name, dataset_name, 'NEAR', sample_name + '.png'),
                mode='GRAY')
            img_B = image_read_cv2(
                os.path.join(img_text_path, 'Image', task_name, dataset_name, 'FAR', sample_name + '.png'),
                mode='GRAY')
            h, w = img_A.shape
            if h < w:
                img_A = cv2.resize(img_A, small_size_0)[None, ...].transpose(1,2,0) / 255.0
                img_B = cv2.resize(img_B, small_size_0)[None, ...].transpose(1,2,0) / 255.0
            else:
                img_A = cv2.resize(img_A, small_size_1).T[None, ...].transpose(1,2,0) / 255.0
                img_B = cv2.resize(img_B, small_size_1).T[None, ...].transpose(1,2,0) / 255.0
            #print(img_A.shape, img_B.shape)
            with open(os.path.join(img_text_path, 'Text', task_name, dataset_name, 'FAR', sample_name + '.txt'), 'r', encoding='utf-8') as file:
                text_array_A = str(file.read())
            with open(os.path.join(img_text_path, 'Text', task_name, dataset_name, 'NEAR', sample_name + '.txt'), 'r', encoding='utf-8') as file:
                text_array_B = str(file.read())
        else:
            img_A = image_read_cv2(
                os.path.join(img_text_path, 'Image', task_name, dataset_name, 'NEAR', sample_name + '.png'),
                mode='GRAY')[None, ...].transpose(1,2,0)/ 255.0
            img_B = image_read_cv2(
                os.path.join(img_text_path, 'Image', task_name, dataset_name, 'FAR', sample_name + '.png'),
                mode='GRAY')[None, ...].transpose(1,2,0)/ 255.0
            with open(os.path.join(img_text_path, 'Text', task_name, dataset_name, 'FAR', sample_name + '.txt'), 'r', encoding='utf-8') as file:
                text_array_A = str(file.read())
            with open(os.path.join(img_text_path, 'Text', task_name, dataset_name, 'NEAR', sample_name + '.txt'), 'r', encoding='utf-8') as file:
                text_array_B = str(file.read())
        imageA.create_dataset(sample_name, data=img_A)
        imageB.create_dataset(sample_name, data=img_B)
        textA.create_dataset(sample_name, data=text_array_A)
        textB.create_dataset(sample_name, data=text_array_B)

