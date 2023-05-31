import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.ops as ops
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

try:
    from config import (DEVICE, TRAIN_DIR, VALID_DIR, OUT_DIR)
    from dataset import create_train_dataset, create_val_dataset
    from model import load_model
except Exception:
    from Object_detection.config import (DEVICE, TRAIN_DIR, VALID_DIR, OUT_DIR)
    from Object_detection.dataset import create_train_dataset, create_val_dataset
    from Object_detection.model import load_model

sns.set()


class Evaluate:
    """
        A class to evaluate the performance of an object detection model on a dataset.

        Attributes:
            model: A PyTorch object detection model that takes in an image and returns a
                dictionary of detections.
            dataloader: A PyTorch DataLoader object that provides a stream of images and
                targets for evaluation.
    """

    def __init__(self, model, dataloader):
        """
        Initializes a new instance of the Evaluate class.
        :param model: A PyTorch object detection model that takes in an image and returns a
                dictionary of detections.
        :param dataloader: A PyTorch DataLoader object that provides a stream of images and
                targets for evaluation.
        """
        self.model = model
        self.dataloader = dataloader

    def evaluation_model_labels(self):
        """
        Evaluates the object detection model on the dataset and returns the predicted and
        ground truth labels for each image.

        :return:  A tuple of two lists: the predicted labels and ground truth labels for each image
            in the dataset. Each list contains a list of labels for each image in the dataset.
        """
        all_predictions_detections = []
        all_targets_detections = []

        with torch.no_grad():
            self.model.eval()
            targets_count = []
            for images, targets in tqdm(self.dataloader):
                # Move the data to the device.
                images = list(image.to(DEVICE) for image in images)
                images = [image.float() / 255.0 for image in images]

                detections = self.model(images)

                boxes_dectection = detections[0]['boxes']
                scores_dectection = detections[0]['scores']
                labels_dectection = detections[0]['labels']

                nms_boxes_dectection = ops.nms(boxes_dectection, scores_dectection, iou_threshold=0.5)

                detections_labels_filtered = labels_dectection[nms_boxes_dectection]
                detections_box_filtered = boxes_dectection[nms_boxes_dectection]

                dectections_final = [{'boxes': detections_box_filtered, 'labels': detections_labels_filtered, }]

                all_predictions_detections.append(dectections_final[0]['labels'])
                all_targets_detections.append(targets[0]['labels'])

        return all_predictions_detections, all_targets_detections


def calculate_acc(predictions_arr, targets_arr):
    """
    Calculates the accuracy of a set of predictions on a set of targets.

    The function compares each predicted label to the corresponding ground truth label in
    the targets array, and counts the number of correct predictions. The accuracy is then
    calculated as the number of correct predictions divided by the total number of predictions.
    :param predictions_arr: A list of predicted labels for a set of images.
    :param targets_arr: A list of ground truth labels for the same set of images.
    :return: The accuracy of the predictions as a float between 0 and 1.

    Raises:
        Exception: If the length of the predictions array is not the same as the length of
            the targets array.
    """
    try:
        if len(predictions_arr) != len(targets_arr):
            raise Exception("Length of predictions array must be as same as with length of targets array")

        same = 0
        for i in range(len(predictions_arr)):
            label_target = targets_arr[i][0]
            labels_prediction = predictions_arr[i].cpu()

            # Count the occurrences of each value in the array.
            counts = np.bincount(labels_prediction)

            if len(counts) != 0:
                # Find the index of the highest count.
                label_prediction = torch.tensor(np.argmax(counts))
            else:
                label_prediction = 0

            if label_target == label_prediction:
                same += 1
        return same / len(predictions_arr)
    except Exception as e:
        print(e)


def learning_curve(start_epochs, end_epoch, model_paths, accuracies_train, accuracies_val, train_dataloader=None,
                   val_dataloader=None):
    """
    Generates a learning curve for a set of models over a range of epochs.

    The function loads a set of pre-trained object detection models from disk, and evaluates
    their performance on a training and/or validation dataset over a specified range of epochs.
    The accuracy of each model is then plotted on a graph, allowing the user to visualize the
    performance of the models over time.
    :param start_epochs: The starting epoch to evaluate the models on.
    :param end_epoch: The ending epoch to evaluate the models on.
    :param model_paths: A list of paths to the model files to evaluate.
    :param accuracies_train: A list to store the accuracy scores for the training dataset.
    :param accuracies_val: A list to store the accuracy scores for the validation dataset.
    :param train_dataloader: A PyTorch DataLoader object that provides a stream of training images
            and targets for evaluation. Defaults to None.
    :param val_dataloader: A PyTorch DataLoader object that provides a stream of validation images
            and targets for evaluation. Defaults to None.
    :return:  None. The function plots the accuracy of each model on the training and validation datasets
        over the specified range of epochs.

    Raises:
        None.
    """
    for i in range(start_epochs, end_epoch + 1):
        path = f"{OUT_DIR}model/model_epoch{i}.pt"  # change this to the actual path of the model file
        model = load_model(path)
        model.to(DEVICE)

        if train_dataloader:
            train_evaluate = Evaluate(model, train_dataloader)
            # Calculate train
            prediction_train, targets_train = train_evaluate.evaluation_model_labels()
            acc_train = calculate_acc(prediction_train, targets_train)
            accuracies_train = np.append(accuracies_train, acc_train)
            # Save the accuracy of each epochs
            np.save(OUT_DIR + 'variable/accuracies_train.npy', accuracies_train)

        if val_dataloader:
            val_evaluate = Evaluate(model, val_dataloader)
            # Calculate val
            predictions_val, targets_val = val_evaluate.evaluation_model_labels()
            acc_val = calculate_acc(predictions_val, targets_val)
            accuracies_val = np.append(accuracies_val, acc_val)
            # Save the accuracy of each epochs
            np.save(OUT_DIR + 'variable/accuracies_val.npy', accuracies_val)

        model_paths = np.append(model_paths, i)
        np.save(OUT_DIR + 'variable/model_paths.npy', model_paths)

    print('After:', model_paths, accuracies_train, accuracies_val)

    # plot the accuracies
    if train_dataloader:
        plt.plot(model_paths, accuracies_train, label='train')

    if val_dataloader:
        plt.plot(model_paths, accuracies_val, label='validation')

    plt.xticks(rotation=90)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Learning curve accuracy")
    plt.legend()
    plt.show()


def __confusion_matrix(targets_arr, prediction_arr):
    """
    Calculates and displays a confusion matrix and classification report for a set of predicted
    and ground truth labels.

    The function calculates a confusion matrix and classification report for the predicted and
    ground truth labels, and displays them in a heatmap and table, respectively. The heatmap shows
    the number of true positives, false positives, true negatives, and false negatives for each
    class, while the classification report shows the precision, recall, F1 score, and support for
    each class.
    :param  targets_arr: A list of ground truth labels for a set of images.
    :param  prediction_arr: A list of predicted labels for the same set of images.
    :return: None. The function displays the confusion matrix and classification report in the console
        and as a heatmap.

    Raises:
        None.
    """
    confusion_matrix_result = confusion_matrix(targets_arr, prediction_arr)

    class_labels = ["__background__", "Babi", "Calimerio", "Chrysanthemum", "Hydrangeas", "Lisianthus", "Pingpong",
                    "Rosy", "Tana"]

    classification_report_result = classification_report(targets_arr, prediction_arr)

    print(classification_report_result)

    labels = class_labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Reds', fmt='.0f', xticklabels=labels, yticklabels=labels)
    plt.title('Flower Recognition')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()


if __name__ == '__main__':
    train_dl = create_train_dataset(TRAIN_DIR)
    val_dl = create_val_dataset(VALID_DIR)
    #     Evaluate
    _path = OUT_DIR + "model/final_model.pt"  # change this to the actual path of the model file
    _model = load_model(_path)
    _model.to(DEVICE)

    final_evaluate = Evaluate(_model, val_dl)

    predictions_final, targets_final = final_evaluate.evaluation_model_labels()

    new_target = []
    new_prediction = []

    for i in range(len(predictions_final)):
        _label_target = targets_final[i][0]
        _labels_prediction = predictions_final[i].cpu()

        # Count the occurrences of each value in the array.
        _counts = np.bincount(_labels_prediction)

        if len(_counts) != 0:
            # Find the index of the highest count.
            label_prediction_best = np.argmax(_counts)
        else:
            label_prediction_best = 0

        new_target.append(_label_target)
        new_prediction.append(label_prediction_best)

    __confusion_matrix(new_target, new_prediction)
