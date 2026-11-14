"""Datasets for zero-shot classification. Returns raw [0, 1] pixel tensors (no CLIP
normalization) so images can go straight into the classifier or through the PGD
attack first.
"""
import json
import os

import torchvision
from torch.utils.data import Dataset

ZERO_SHOT_TEMPLATE = "a photo of a {}."


class CIFARZeroShot(Dataset):
    """Quick sanity-check dataset: small, auto-downloads. Use CIFAR-100 to validate the
    pipeline runs end-to-end before pointing it at the real ImageNet-100 data; CIFAR-10
    serves as a second, held-out dataset for checking that clean capability is retained.
    """

    def __init__(self, root: str, train: bool, resize_crop_transform, num_classes: int = 100):
        cls = torchvision.datasets.CIFAR100 if num_classes == 100 else torchvision.datasets.CIFAR10
        self.base = cls(root=root, train=train, download=True)
        self.transform = resize_crop_transform

    @property
    def classnames(self):
        # "maple_tree" -> "maple tree", matching the prompts used in PaperThree-Five
        return [c.replace("_", " ") for c in self.base.classes]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


class ImageFolderZeroShot(Dataset):
    """For the real experiment: ImageNet-100 laid out as root/<class_folder>/*.jpg.

    Pass a classnames_json mapping folder name -> human-readable class name (e.g.
    ImageNet's synset id -> label text) if folder names aren't already readable;
    otherwise folder names are used as-is.
    """

    def __init__(self, root: str, resize_crop_transform, classnames_json: str = None):
        self.base = torchvision.datasets.ImageFolder(root=root)
        self.transform = resize_crop_transform
        folder_names = self.base.classes
        if classnames_json and os.path.exists(classnames_json):
            with open(classnames_json) as f:
                mapping = json.load(f)
            self._classnames = [mapping.get(c, c) for c in folder_names]
        else:
            self._classnames = folder_names

    @property
    def classnames(self):
        return self._classnames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img.convert("RGB")), label


def build_dataset(name: str, root: str, train: bool, resize_crop_transform, classnames_json: str = None):
    if name == "cifar100":
        return CIFARZeroShot(root, train, resize_crop_transform, num_classes=100)
    if name == "cifar10":
        return CIFARZeroShot(root, train, resize_crop_transform, num_classes=10)
    if name == "imagefolder":
        return ImageFolderZeroShot(root, resize_crop_transform, classnames_json)
    raise ValueError(f"unknown dataset {name}")
