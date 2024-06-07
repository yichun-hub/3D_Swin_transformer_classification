'''
Custom Dataloader
'''

import math
import os

import numpy as np
import torch
import random

from monai import data, transforms
from monai.data import load_decathlon_datalist

from torch.utils.data import DataLoader
import torchio as tio


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, data_aug = False, transform = None):
        self.file_list = file_list
        self.transform = transform
        self.class_folders = ['TP', 'FP']
        self.data_aug = data_aug
        self.image, self.label = self._generate_preprocessed_series_data() # 要注意這種讀取方式會佔記憶體
        

    def _generate_preprocessed_series_data(self):

        x = []
        y = []
        
        # Read npz file in splited series_uid list
        for file_path in self.file_list:
            #print(file_path)
            if "TP" in file_path:
                label = "TP"
                y.append(0)
            elif "FP" in file_path:
                label = "FP"
                y.append(1)              
            
            # Load image
            f = np.load(file_path)
            #print(f"{series_npz_path}")
            image = f['image']   
            image = np.expand_dims(image, axis=0).astype(np.float32)     

            #print(image.min(), image.max()) 0,1
            x.append(image)

        x = np.asarray(x, np.float32)
        y = np.asarray(y)


        if self.data_aug:
            x, y = self.data_augment(x, y, mode='img2label')


        return x, y
    
    def data_augmentation_3d_array(self, arr):
        buff = []

        # Head
        # Extend rotations of original image and reflectional image
        ref = np.flip(arr, 1)
        buff.extend([arr, ref]) # 0
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 90
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 180
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 270
        arr = np.rot90(arr, axes=(1,2))

        # Flipping
        arr = np.flip(arr, axis=(0,1))
        ref = np.flip(arr, 1)

        # Tail
        # Extend rotations of original image and reflectional image
        buff.extend([arr, ref]) # 0
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 90
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 180
        arr = np.rot90(arr, axes=(1,2))
        ref = np.rot90(ref, axes=(1,2))
        buff.extend([arr, ref]) # 270

        return buff

    def data_augment(self, x, y, mode='img2label', seed=1):

        # Generate file id
        file_id = list(range(0, len(x)*16))
        random.Random(seed).shuffle(file_id)

        # Augment 3D array 
        x_aug = []
        y_aug = []
        for i in range(len(x)): #2 nodules
            x_buff = self.data_augmentation_3d_array(x[i])
            if mode == 'img2label':
                #print(y[i])
                y_buff = [y[i] for _ in range(16)]
            elif mode == 'img2img':
                y_buff = self.data_augmentation_3d_array(y[i])
            for j in range(len(x_buff)): #16            
                x_aug.append(x_buff[j])
                y_aug.append(y_buff[j])
                del file_id[0]
        #x_aug = np.asarray(x_aug)
        #y_aug = np.asarray(y_aug)
        return x_aug, y_aug


    def __len__(self):
        return len(self.image) ## total length of augment image

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]

        # transform
        if self.transform:
            image = self.transform(image)

        # Data augmentation 
        # if self.data_aug:
        #     image, label = self.data_augment(image, label, mode='img2label')

        return torch.tensor(image, dtype=torch.float), label



class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def data_series_list(series_npz_path):
    all_series = []
    for contrast_type in os.listdir(series_npz_path):
        if contrast_type.endswith('.sh') or contrast_type.endswith('.txt'):
            continue
        contrast_type_path = os.path.join(series_npz_path, contrast_type)
        for texture in os.listdir(contrast_type_path):
            texture_path = os.path.join(contrast_type_path, texture) 
            for file in os.listdir(texture_path):  
                file_path = os.path.join(texture_path, file)
                all_series.append(file_path)

    return all_series 

def FP_data_series_list(series_npz_path):

    all_series = []
    for batch in os.listdir(series_npz_path):
        batch_path = os.path.join(series_npz_path, batch)
        for file in os.listdir(batch_path):  
            file_path = os.path.join(batch_path, file)
            all_series.append(file_path)
    return all_series     

def vesseldata_series_list(series_npz_path):
    all_series = []
    for file in os.listdir(series_npz_path): 
        file_path = os.path.join(series_npz_path, file)
        all_series.append(file_path)

    return all_series    


def Testing_data_series_list(series_npz_path):
    all_series = []
    for type in os.listdir(series_npz_path):
        if type.endswith('.sh') or type.endswith('.txt'):
            continue

        type_path = os.path.join(series_npz_path, type)

        for file in os.listdir(type_path):  
            file_path = os.path.join(type_path, file)
            all_series.append(file_path)

    return all_series

def get_training_validation_set_series(series_list, porpotion, seed=1):
    import random
    series_uid = sorted(series_list)
    random.Random(seed).shuffle(series_uid)
    num = len(series_uid)
    train = series_uid[:int(num*porpotion[0])]
    validation = series_uid[int(num*porpotion[0]):int(num*(porpotion[0]+porpotion[1]))]
    test = series_uid[int(num*(porpotion[0]+porpotion[1])):]
    return train, validation, test

def Custom_loader(args, tp_dir, fp_dir, vessel_fp_dir):
    TP_series_list = FP_data_series_list(tp_dir)
    #FP_series_list = data_series_list(fp_dir)
    FP_series_list = FP_data_series_list(fp_dir)
    #print(FP_series_list)
    vessel_fp_series_list = vesseldata_series_list(vessel_fp_dir)
    All_FP_series_list = FP_series_list + vessel_fp_series_list


    train_tp, validation_tp, test_tp =  get_training_validation_set_series(TP_series_list, porpotion= [0.8, 0.2, 0])
    train_fp, validation_fp, test_fp =  get_training_validation_set_series(All_FP_series_list, porpotion= [0.8, 0.2, 0])
    
    print('Numbers of Train TP', len(train_tp))
    print('Numbers of Train FP', len(train_fp))
    print('Numbers of Val TP', len(validation_tp))
    print('Numbers of Val FP', len(validation_fp))
    
    training_data_list = train_tp + train_fp
    validation_data_list = validation_tp + validation_fp

    random.shuffle(training_data_list)
    random.shuffle(validation_data_list)

    # transformer
    RandomFlip = tio.RandomFlip()
    train_transform = tio.transforms.Compose(
        [
            RandomFlip

            ]
            )
    val_transform = tio.transforms.Compose(
        [
            RandomFlip
            ]
            )

    train_dataset = CustomDataset(training_data_list, data_aug = True, transform=train_transform)
    valid_datset = CustomDataset(validation_data_list, data_aug = False, transform=None)

    print('Numbers of Training data: ', len(train_dataset))
    print('Numbers of Validation data: ',len(valid_datset))

    train_sampler = Sampler(train_dataset) if args.distributed else None
    val_sampler = Sampler(valid_datset, shuffle=False) if args.distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_datset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, pin_memory=True)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    loader = [train_loader, valid_loader]

    return loader


def test_custom_loader(args, test_dir):
    test_series_list = Testing_data_series_list(test_dir)
    train, validation, test =  get_training_validation_set_series(test_series_list, porpotion= [0, 0, 1.0])
    print('Numbers of Testing data', len(test))

    test_dataset = CustomDataset(test, data_aug = False)
    test_sampler = Sampler(test_dataset, shuffle=False) if args.distributed else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=test_sampler, pin_memory=True)

    return test_loader

