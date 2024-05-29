import os
from glob import glob
from typing import Dict
from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np

class DefaultDataset(VisionDataset):
    """ Dataset.

    Args:
        root (string): directory of original dataset containing images
    """

    def __init__(
            self,
            root: str,
    ) -> None:
        super(DefaultDataset, self).__init__(root)
        self.samples = []

        if not os.path.exists(root):
            print(f"Path to root directory not valid: {root}")

        filepaths = glob(os.path.join(root, "*.png"))
        if len(filepaths) == 0:
            filepaths = glob(os.path.join(root, "*.jpg"))

        for file in filepaths:
            if "_mask." in file:
                continue
            imageName = os.path.basename(file).split(".")[0]
            imagePath = file
            sample = {
                'imageName'    : imageName,
                'imagePath'    : imagePath,
            }
            self.samples.append(sample)
        return


    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index (int): Index
        Returns:
            dict
        """
        sample = self.samples[index]

        if not 'image' in sample:
            sample['image'] = Image.open(self.samples[index]['imagePath'])
            img_np = np.asarray(sample['image'])
            img_np = np.einsum("kij->jki", img_np)
            sample['image'] = img_np/255.0

        return sample


    def __len__(self) -> int:
        return len(self.samples)
