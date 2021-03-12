# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import glob
import random
import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
mean = [0.5]
std = [0.5]

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),10)
    new_image = image_blur
    return new_image


class Dataset_(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.functional.to_grayscale,
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".avi")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name)]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Pad frames sequences shorter than `self.sequence_length` to length
#        if len(image_paths) ==0:
#            print(sequence_path)
#            print(index)
        image_paths = self._pad_to_length(image_paths)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        print(len(image_sequence));
        print(type(image_sequence[0]));
        print(image_sequence[0].shape);
        image_sequence = torch.stack(image_sequence)
        target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)




class Dataset_prediction(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length+1;
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.functional.to_grayscale,
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".avi")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name)]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Pad frames sequences shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        target = image_sequence[-1];
        image_sequence = image_sequence[:-1];
        image_sequence = torch.stack(image_sequence);
        image_sequence += 0.1*torch.rand_like(image_sequence);
        #target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)





class Dataset_prediction_full(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length+1;
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)
#        self.transform = transforms.Compose(
#            [
#                transforms.functional.to_grayscale,
#                transforms.Resize(input_shape[-2:], Image.BICUBIC),
#                transforms.Lambda(gaussian_blur),
#                transforms.ToTensor(),
#                transforms.Normalize(mean, std),
#            ]
#        )
        self.transform_first = transforms.Compose(
            [
                transforms.functional.to_grayscale,
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
            ]
        )

        self.transform_bis = transforms.Compose(
            [
                transforms.Lambda(gaussian_blur),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_second = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )



    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".avi")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name)]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Pad frames sequences shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        output_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor  = self.transform_first(Image.open(image_paths[i]))
                output_tensor = self.transform_second(image_tensor);
                image_tensor  = self.transform_bis(image_tensor);
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                    output_tensor = torch.flip(output_tensor, (-1,))
                image_sequence.append(image_tensor)
                output_sequence.append(output_tensor)
        target = output_sequence[1:];
        target = torch.stack(target);
        image_sequence = image_sequence[:-1];
        image_sequence = torch.stack(image_sequence);
        #image_sequence += 0.1*torch.rand_like(image_sequence);
        #target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
