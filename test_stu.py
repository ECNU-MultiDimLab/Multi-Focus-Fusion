import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
from utils.Evaluator import Evaluator
from utils.H5_read import H5ImageTextDataset
import warnings
from torch.utils.data import Dataset,DataLoader
import argparse
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
task_name = 'MFF'
dataset_name = 'Pathology'
if dataset_name == 'Lytro' or dataset_name == 'RealMFF':
    ckpt_path = os.path.join('checkpoints','stu.pth')
    from net.Fuse_Net_stu import Net
elif dataset_name == 'Pathology':
    ckpt_path = os.path.join('checkpoints', 'stu_path.pth')
    from net.Fuse_Net_stu_path import Net
test_folder = os.path.join('VLFDataset','Image','MFF',dataset_name)
dataset_path = os.path.join('VLFDataset_h5', dataset_name + '_test.h5')
testloader = DataLoader(H5ImageTextDataset(dataset_path), batch_size=1,
                         shuffle=True, num_workers=0)


save_path = os.path.join("test_output", dataset_name, "stu")
os.makedirs(save_path, exist_ok=True)

device = 'cuda'
Net = Net().to(device)
checkpoint = torch.load(ckpt_path)
state_dict = checkpoint['model']
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
Net.load_state_dict(new_state_dict)
Net.eval()
test_index = []
with torch.no_grad():
    for i, (imageA, imageB, text_A, text_B, index) in enumerate(tqdm(testloader)):
        imageA = imageA.to('cuda')
        imageB = imageB.to('cuda')
        Fuse_image, _, _, _, _, _ = Net(imageA, imageB)
        Fuse_image = (Fuse_image - torch.min(Fuse_image)) / (torch.max(Fuse_image) - torch.min(Fuse_image))
        fi = np.squeeze((Fuse_image * 255).detach().cpu().numpy())
        fi = fi.astype('uint8')
        index = index[0]
        test_index.append(index)
        img_save(fi, index, save_path)

eval_folder=save_path
ori_img_folder=test_folder

metric_result = np.zeros((5))
for i in tqdm(test_index):
    image_A = image_read_cv2(os.path.join(ori_img_folder,"FAR", str(i) + ".png"), 'GRAY')
    image_B = image_read_cv2(os.path.join(ori_img_folder,"NEAR", str(i) + ".png"), 'GRAY')
    fi = image_read_cv2(os.path.join(eval_folder, str(i) + ".png"), 'GRAY')
    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                , Evaluator.SF(fi)
                                , Evaluator.VIFF(fi, image_A, image_B)
                                , Evaluator.Qabf(fi, image_A, image_B)])
metric_result /= len(test_index)
print("EN\t SD\t SF\t VIF\t Qabf")
print(str(np.round(metric_result[0], 2))+'\t'
        +str(np.round(metric_result[1], 2))+'\t'
        +str(np.round(metric_result[2], 2))+'\t'
        +str(np.round(metric_result[3], 2))+'\t'
        +str(np.round(metric_result[4], 2)))
print("="*80)