"""
overwrites the MONAI 0.2.0 ArrayDataset class
returns the ID of a case in addition to the input and ground truth
"""



import hashlib
import json
import sys
import threading
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset

from monai.transforms import apply_transform, Compose, Randomizable, Transform
from monai.utils import get_seed, progress_bar


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data, transform: Optional[Callable] = None):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class ZipDataset(Dataset):
    """
    Zip several PyTorch datasets and output data(with the same index) together in a tuple.
    If the output of single dataset is already a tuple, flatten it and extend to the result.
    For example: if datasetA returns (img, imgmeta), datasetB returns (seg, segmeta),
    finally return (img, imgmeta, seg, segmeta).
    And if the datasets don't have same length, use the minimum length of them as the length
    of ZipDataset.

    Examples::

        >>> zip_data = ZipDataset([[1, 2, 3], [4, 5]])
        >>> print(len(zip_data))
        2
        >>> for item in zip_data:
        >>>    print(item)
        [1, 4]
        [2, 5]

    """

    def __init__(self, datasets, transform: Optional[Callable] = None):
        """
        Args:
            datasets (list or tuple): list of datasets to zip together.
            transform: a callable data transform operates on the zipped item from `datasets`.
        """
        super().__init__(list(datasets), transform=transform)

    def __len__(self):
        return min([len(dataset) for dataset in self.data])

    def __getitem__(self, index: int):
        def to_list(x):
            return list(x) if isinstance(x, (tuple, list)) else [x]

        data = list()
        for dataset in self.data:
            data.extend(to_list(dataset[index]))
        if self.transform is not None:
            data = apply_transform(self.transform, data, map_items=False)  # transform the list data
        return data



class ArrayDataset(Randomizable, _TorchDataset):
    # overwrites MONAI ArrayDataset class

    def __init__(
        self,
        img,
        img_transform: Optional[Callable] = None,
        seg=None,
        seg_transform: Optional[Callable] = None,
        labels=None,
        label_transform: Optional[Callable] = None,
    ):
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img (Sequence): sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg (Sequence, optional): sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels (Sequence, optional): sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed
        
        self.names = [path[-9:-4] for path in img]

    def __len__(self):
        return len(self.dataset)
    
    
    def randomize(self):
        self._seed = self.R.randint(np.iinfo(np.int32).max)


    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index], self.names[index]