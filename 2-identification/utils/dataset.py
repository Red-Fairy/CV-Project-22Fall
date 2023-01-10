from PIL import Image
import os
import os.path
import numpy as np
import torchvision.transforms as transforms
import torch
import copy

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import extract_archive

import random

def cutout(img, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            if erase_w < img_w and erase_h < img_h:
                x = np.random.randint(0, img_w - erase_w)
                y = np.random.randint(0, img_h - erase_h)
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w, :] = value

        img = Image.fromarray(img.astype(np.uint8))

    return img

CLASS = {0: 'Azibo', 1: 'Natascha', 2: 'Ohini', 3: 'Swela', 4: 'Frodo', 
         5: 'Dorien', 6: 'Lome', 7: 'Lobo', 8: 'Kisha', 
         9: 'Fraukje', 10: 'Riet', 11: 'Sandra', 12: 'Kofi', 
         13: 'Bambari', 14: 'Tai', 15: 'Corrie', 16: 'Maja'}

# CLASS = {0: 'Azibo', 1: 'Natascha', 2: 'Ohini', 3: 'Swela', 
#         4: 'Frodo', 5: 'Dorien', 6: 'Lome', 7: 'Lobo', 
#         8: 'Kisha', 9: 'Fraukje', 10: 'Riet', 11: 'Sandra', 12: 'Bambari'}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ChimpDataset(VisionDataset):
    """`CODaN <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        data (string, optional): Location of the downloaded .tar.bz2 files.
        split (string, optional): Define which dataset split to use. Must be 'train' or 'val'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root='./', split='train', transform=None, target_transform=None, 
                contrastive=False, transform_contrastive=None, 
                mixup=False, cutout=False, BCL=False):

        super(ChimpDataset, self).__init__(root, transform, target_transform)

        self.split = split  # dataset split
        self.data = []
        self.targets = []
        self.paths = []
        self.contrastive = contrastive
        self.transform_contrastive = transform_contrastive
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.transform = transform
        self.target_transform = target_transform
        self.mixup = mixup
        self.cutout = cutout
        self.BCL = BCL

        # loop through split directory, load all images in memory using PIL
        for i, c in enumerate(CLASS):
            im_dir = os.path.join(root,'data/images_id_individual',split,CLASS[c])
            try:
                ims = os.listdir(im_dir)
            except: # Some classes are missing in the validation set
                continue
            ims = [im for im in ims if is_image_file(im)] # remove any system files
            
            for im in ims:
                img = Image.open(os.path.join(im_dir,im))
                self.data.append(img.copy())
                img.close()
                self.paths.append(os.path.join(im_dir,im))
                self.targets.append(i)
        print('Dataset {} split loaded.'.format(split), 'Number of images:', len(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.cutout:
            img = cutout(img)

        if self.mixup:
            # random sample another image
            index2 = np.random.randint(0, len(self.data))
            img2, target2 = self.data[index2], self.targets[index2]
            # transform both images
            img = self.transform(img)
            img2 = self.transform(img2)
            # mixup
            lam = np.random.beta(1.0, 1.0)
            img = lam * img + (1 - lam) * img2
            # transform target to one-hot
            t = torch.zeros(len(CLASS))
            t[target2] = 1 - lam
            t[target] = lam
            return img, t

        if not self.contrastive:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        
        else:

            img2 = copy.deepcopy(img)
            img3 = copy.deepcopy(img)

            img = self.transform(img)
            img2 = self.transform_contrastive(img2)

            img3 = self.transform_contrastive(img3)
            
            if self.BCL:
                return [img, img2, img3], target

            else:
                return torch.stack([img2, img3]), img, target
        


    def __len__(self):
        return len(self.data)


