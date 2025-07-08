import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化时预处理所有图像
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.preprocessed_data = {}
        
        print(f"开始加载数据集: {root_dir}")
        # 收集并预处理所有样本
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        # 加载并预处理图像
                        image = Image.open(image_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                            if isinstance(image, torch.Tensor):
                                # 将tensor移到CPU并固定内存
                                image = image.cpu().pin_memory()
                        
                        # 存储预处理后的图像和标签
                        self.preprocessed_data[image_path] = {
                            'image': image,
                            'label': label
                        }
                        self.samples.append(image_path)
                        
                        if len(self.samples) % 100 == 0:
                            print(f"已加载 {len(self.samples)} 张图片")
                            
                    except Exception as e:
                        print(f"加载图片失败 {image_path}: {e}")
        
        print(f"数据集加载完成，共 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        直接返回预处理好的数据
        """
        image_path = self.samples[idx]
        data = self.preprocessed_data[image_path]
        return data['image'], data['label'], image_path
