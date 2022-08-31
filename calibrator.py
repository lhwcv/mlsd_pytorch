import os

import cv2
import numpy as np

import torch
from torchvision.datasets import ImageFolder

class ImageFolderCalibDataset():

    def __init__(self, root):
        self.dataset = ImageFolder(
            root=root
        )
        self.input_shape=[512, 512]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = np.asarray(image)
        img = cv2.resize(image, (512, 512))
        resized_image = np.concatenate([cv2.resize(image, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_AREA),
                                        np.ones([self.input_shape[0], self.input_shape[1], 1])], axis=-1)

        resized_image = resized_image.transpose((2,0,1))
        batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
        batch_image = (batch_image / 127.5) - 1.0
        batch_image = torch.from_numpy(batch_image).float().cuda()
        
        return batch_image

