import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Process all images in initialize
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.preprocessed_data = {}
        
        print(f"Starting load Dataset: {root_dir}")

        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                            if isinstance(image, torch.Tensor):
                                # 将tensor移到CPU并固定内存
                                image = image.cpu().pin_memory()
                        
                        self.preprocessed_data[image_path] = {
                            'image': image,
                            'label': label
                        }
                        self.samples.append(image_path)
                        
                        if len(self.samples) % 100 == 0:
                            print(f"Already loaded {len(self.samples)} images")
                            
                    except Exception as e:
                        print(f"fail to load {image_path}: {e}")
        
        print(f"Dataset load complete all {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        data = self.preprocessed_data[image_path]
        return data['image'], data['label'], image_path
