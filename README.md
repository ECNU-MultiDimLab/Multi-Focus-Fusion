# Mg-FILM: Multi-grained feature alignment and Fusion based on distillation and vIsion-Language Model
Code and dataset for ***Text-Guided Defocus Knowledge Distillation for Multi-Focus Image Fusion***
### Testing

**1. Pretrained models**

The pre-trained models can be found at ``'./checkpoints/stu.pth'`` and ``'./checkpoints/stu_path.pth'``. The first model is responsible for RealMFF and Lytro datasets and the second model is responsible for Pancreatic Cancer dataset.

**2. Test datasets**

The images of the test dataset used in the paper were provided in the format ``'./VLFDataset/Image/MFF/{Dataset_name}'``.
The texts of the test dataset used in the paper were provided in the format ``'./VLFDataset/Text/MFF/{Dataset_name}'``.

**3. Pre-Processing**

Run 
```
python data_process.py
``` 
and the processed training dataset is in ``'./VLFDataset_h5/'``.

**4. Results in Our Paper**

If you want to infer with our Mg-FILM and obtain the fusion results in our paper, please run 
```
python test_stu.py
``` 
to perform image fusion. Set the variable "dataset_name" to the name of the dataset that you want to infer. The output fusion results will be saved in the ``'./test_output/{Dataset_name}/stu'``  folder.
