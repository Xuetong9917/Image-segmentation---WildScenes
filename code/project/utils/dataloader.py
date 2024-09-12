import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.initialization import cvtColor, preprocess_input

class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        
        # Load images and labels
        jpg = Image.open(os.path.join(self.dataset_path, "JPEGImages", f"{name}.jpg"))
        png = Image.open(os.path.join(self.dataset_path, "SegmentationClass", f"{name}.png"))
        
        # Preprocess the images and labels
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        # Prepare input and label tensors
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((self.input_shape[0], self.input_shape[1], self.num_classes + 1))

        return jpg, png, seg_labels

    @staticmethod
    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # Resize and augment the image and label
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # Resize image and label without augmentation
            scale = min(w/iw, h/ih)
            nw, nh = int(iw*scale), int(ih*scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), 0)
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # Apply random augmentations
        new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # Random flipping
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # Random placement
        dx, dy = int(self.rand(0, w - nw)), int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))

        image = new_image
        label = new_label

        # Convert to numpy array
        image_data = np.array(image, np.uint8)

        # Apply Gaussian blur
        if self.rand() < 0.25:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # Random rotation
        if self.rand() < 0.25:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        # Color jittering
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        lut_hue = ((np.arange(0, 256, dtype=r.dtype) * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(np.arange(0, 256, dtype=r.dtype) * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(np.arange(0, 256, dtype=r.dtype) * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

def deeplab_dataset_collate(batch):
    images, pngs, seg_labels = zip(*batch)
    images = torch.from_numpy(np.array(images)).float()
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()
    return images, pngs, seg_labels
