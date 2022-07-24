import paddle
from paddle.io import Dataset
from PIL import Image
import os
import numpy as np


def read_path_DUTS(path, split='train', edge=True):
    if split == 'train':
        image_root = os.path.join(path, 'DUTS-TR', 'DUTS-TR-Image')
        mask_root = os.path.join(path, 'DUTS-TR', 'DUTS-TR-Mask')
        if edge:
            edge_root = os.path.join(path, 'DUTS-TR', 'DUTS-TR-Edge')
    else:
        image_root = os.path.join(path, 'DUTS-TE', 'DUTS-TE-Image')
        mask_root = os.path.join(path, 'DUTS-TE', 'DUTS-TE-Mask')
        if edge:
            edge_root = os.path.join(path, 'DUTS-TE', 'DUTS-TE-Edge')

    image_names = os.listdir(image_root)
    image_paths = [os.path.join(image_root, n) for n in image_names]
    mask_paths = [os.path.join(mask_root, n.split('.')[0]+'.png') for n in image_names]

    if edge:
        edge_paths = [os.path.join(edge_root, n.split('.')[0]+'.png') for n in image_names]
        return image_paths, mask_paths, edge_paths
    return image_paths, mask_paths 

def read_path_test(path, name: str):
    folder_key = {
        'ECSSD': ['Imgs', 'Gt'], 'SOD': ['images', 'gt'],
        'PASCAL-S': ['Imgs', 'Gt'], 'HKU-IS': ['imgs', 'gt'],
        'DUT-OMRON': ['Imgs', 'Gt'], 'DUTS-TE': ['DUTS-TE-Image', 'DUTS-TE-Mask']
    }

    image_root = os.path.join(path, folder_key[name][0])
    mask_root = os.path.join(path, folder_key[name][1])

    image_names = os.listdir(image_root)
    image_paths = [os.path.join(image_root, n) for n in image_names]
    mask_paths = [os.path.join(mask_root, n.split('.')[0]+'.png') for n in image_names]

    return image_names, image_paths, mask_paths


class DUTS(Dataset):
    def __init__(self, path="./dataset/DUTS", split='train'):
        super().__init__()

        self.path = path
        self.split = split
        self.image_list, self.mask_list, self.edge_list = read_path_DUTS(path, split)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        gt = Image.open(self.mask_list[idx])
        edge = Image.open(self.edge_list[idx])
        img = img.resize((256, 256))
        gt = gt.resize((256, 256))
        # to numpy array and normalize
        img_arr = np.array(img).transpose(2, 0, 1) / 255
        img_arr = (img_arr - 0.5) / 0.5
        gt_arr = np.array(gt) / 255
        edge_arr = np.array(edge) / 255
        if len(gt_arr.shape) != 2:
            gt_arr = gt_arr[:, :, 0]
        if len(edge_arr.shape) != 2:
           edge_arr = edge_arr[:, :, 0]
        # random flip
        choice = np.random.choice([0, 1])
        if choice == 1:
            img_arr = img_arr[:, :, ::-1]
            gt_arr = gt_arr[:, ::-1]
            edge_arr = edge_arr[:, ::-1]
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        edge_tensor = paddle.to_tensor(edge_arr).astype('float32')
        return img_tensor, gt_tensor, edge_tensor

    def __len__(self):
        return len(self.image_list)


class DATASET_TEST(Dataset):
    def __init__(self, path, name, return_name=False):
        super().__init__()
        self.path = path
        self.name = name
        self.return_name = return_name
        self.name_list, self.image_list, self.mask_list = read_path_test(path, name)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        gt = Image.open(self.mask_list[idx])
        h, w = img.size[1], img.size[0]
        img = img.resize((256, 256))
        # to numpy array and normalize
        if len(np.array(img).shape) == 3:
            img_arr = np.array(img).transpose(2, 0, 1) / 255
        else:
            _img_arr = np.array(img).reshape((1, 256, 256))
            img_arr = np.concatenate([_img_arr, _img_arr, _img_arr], axis=0) / 255
        img_arr = (img_arr-0.5) / 0.5
        gt_arr = np.array(gt) / 255
        # to paddle tensor
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        if len(gt_tensor.shape) != 2:
            gt_tensor = gt_tensor[:, :, 0]
        # img, label, h, w
        if not self.return_name:
            return img_tensor, gt_tensor, h, w
        else:
            return img_tensor, gt_tensor, h, w, self.name_list[idx]

    def __len__(self):
        return len(self.image_list)

