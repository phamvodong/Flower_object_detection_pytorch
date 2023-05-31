import numpy as np
import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from Object_detection.config import (BATCH_SIZE, NUM_WORKERS, LABEL_MAP, TRAIN_DIR)
    from Object_detection.custom_utils import collate_fn, get_transforms
    from Object_detection.custom_utils import view
except Exception:
    from config import (BATCH_SIZE, NUM_WORKERS, LABEL_MAP, TRAIN_DIR)
    from custom_utils import collate_fn, get_transforms
    from custom_utils import view


class CustomDataset(Dataset):
    def __init__(self, df, label_map, transforms=None):
        """
        A PyTorch dataset class for the flower detection task.
        :param df : pandas.DataFrame
        A Pandas DataFrame containing the file path, bounding boxes, and class labels for each image.
        :param label_map : dict
        A dictionary mapping class label strings to integer indices.
        :param transforms : albumentations.Compose
        A set of image augmentations to apply to each input image and corresponding bounding box.
        """
        # Read the CSV file into a Pandas DataFrame.
        self.df = df

        # Create a list of image paths and bounding boxes.
        self.images = self.df['path'].tolist()

        # Create a map from flower type to label.
        self.label_map = label_map

        # Transforms the image
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get the image path and bounding box from the index.
        image_path = self.images[index]
        bounding_boxes = self.df[self.df.path == image_path].values[:, 2:].astype('float')
        flower_types = self.df[self.df.path == image_path]['flower'].tolist()

        # Convert the flower types into labels.
        labels = [self.label_map[flower_type] for flower_type in flower_types]

        # Load the image.
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)  # Convert the PIL.Image object to a NumPy array.

        # Resize the image and transform the bounding boxes.
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bounding_boxes, labels=labels)
            image = transformed['image']
            bounding_boxes = transformed['bboxes']
            labels = transformed['labels']

        # Create the target dictionary.
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            'boxes': torch.tensor(bounding_boxes),
            'labels': labels
        }

        return image, target


# Prepare the datasets and data loaders
def create_train_dataset(file_name):
    """
    Creates a PyTorch DataLoader for the training dataset.

    :param  file_name : str
        The name of the CSV file containing the training dataset.
    :return: train_dataloader : torch.utils.data.DataLoader
        A PyTorch DataLoader containing the training dataset.
    """
    train_df = pandas.read_csv(file_name)
    train_dataloader = DataLoader(
        CustomDataset(train_df, LABEL_MAP, get_transforms(True)),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_dataloader


def create_val_dataset(file_name):
    """
    Creates a PyTorch DataLoader for the validation dataset.

    :param file_name : str
        The name of the CSV file containing the validation dataset.

    :return: val_dataloader : torch.utils.data.DataLoader
        A PyTorch DataLoader containing the validation dataset.
    """
    val_df = pandas.read_csv(file_name)
    val_dataloader = DataLoader(
        CustomDataset(val_df, LABEL_MAP, get_transforms(False)),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return val_dataloader


if __name__ == '__main__':
    train_dl = create_train_dataset(TRAIN_DIR)

    images, labels = next(iter(train_dl))

    view(images, labels, BATCH_SIZE, label_map=LABEL_MAP)
