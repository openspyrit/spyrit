from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class NetImage(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'imagenet-batches'
    train_list = [
        ['train_data_batch_1', '7d78180ed6d675199904d73e97363aa3'],
        ['train_data_batch_2', '62979cbd524679ea440f2eb998cf70ed'],
        ['train_data_batch_3', '022d13e31ebf76e3a3b4995f59d5898b'],
    ]

    test_list = [
        ['val_data', '68a29f115231937c359924d8af1b0922'],
    ]
#    meta = {
#        'filename': 'batches.meta',
#        'key': 'label_names',
#        'md5': '5ff9c542aee3614f3951f8cda6e48888',
#    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super(NetImage, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

       
        self.train = train 



        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

#        self._load_meta()

#    def _load_meta(self) -> None:
#        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#        if not check_integrity(path, self.meta['md5']):
#            raise RuntimeError('Dataset metadata file not found or corrupted.' +
#                               ' You can use download=True to download it')
#        with open(path, 'rb') as infile:
#            data = pickle.load(infile, encoding='latin1')
#            self.classes = data[self.meta['key']]
#        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
