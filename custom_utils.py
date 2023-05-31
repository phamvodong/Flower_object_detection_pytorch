import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(train=False):
    """
    Returns a set of image augmentations as an Albumentations Compose object.
    :param train : bool
        If True, apply augmentations suitable for training data, else use minimal augmentations
        suitable for validation/test data.
    :return:
      transforms : albumentations.Compose
        An Albumentations Compose object containing image augmentations.
    """
    if train:
        transforms = A.Compose([
            A.Resize(height=250, width=250),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transforms = A.Compose([
            A.Resize(250, 250),  # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return transforms


def get_detection_transforms():
    """
    Returns a set of image transforms for use with object detection tasks.

    The transforms include resizing the image to a fixed size of 250x250 pixels, and
    converting the image to a PyTorch tensor.

    :return:
     A composition of image transforms that can be applied to an image, using the
        Albumentations library.
    """
    transforms = A.Compose([
        A.Resize(250, 250),
        ToTensorV2()
    ])

    return transforms


def collate_fn(batch):
    """
    A collate function used to merge a list of samples into a batch.
    :param batch : list of tuples
        A list of tuples, where each tuple contains the elements of a single sample.
    :return: tuple
        A tuple of lists, where each list contains the elements of a batched sample.
    """
    return tuple(zip(*batch))


def view(images, labels, k, std=1, mean=0, label_map=None):
    """
    Displays a grid of images with their corresponding bounding boxes and labels.
    :param images: torch.Tensor
        A tensor of shape (batch_size, channels, height, width) containing input images.
    :param labels: list
        A list of dictionaries, where each dictionary contains 'boxes' and 'labels' keys
        representing the bounding boxes and class labels for each image in the batch.
    :param k: int
        The number of images to display in the grid.
    :param std: float, optional
        The standard deviation used for input image normalization. Default is 1.
    :param mean: float, optional
        The mean used for input image normalization. Default is 0.
    :param label_map: dict, optional
        A dictionary mapping class label indices to corresponding string names. Default is None.
    :return: None
    """
    figure = plt.figure(figsize=(30, 30))
    images = list(images)
    labels = list(labels)
    for i in range(k):
        out = torchvision.utils.make_grid(images[i])
        inp = out.cpu().numpy().transpose((1, 2, 0))
        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)
        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        ax.axis('off')
        bboxes = labels[i]['boxes'].cpu().detach().numpy()
        bboxes = bboxes.reshape(-1, 4)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        labels_idx = labels[i]['labels'].cpu().numpy()
        if isinstance(labels_idx, int):
            labels_idx = np.array([labels_idx])
        labels_str = [list(label_map.keys())[list(label_map.values()).index(idx)] for idx in labels_idx]
        for j in range(len(bboxes)):
            label = labels_str[j]
            bbox = bboxes[j]
            ax.add_patch(
                patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='w', facecolor='none'))
            ax.text(bbox[0], bbox[1], label, fontsize=12, color='r')
    plt.show()
