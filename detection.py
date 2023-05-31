import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image

try:
    from Object_detection.config import (LABEL_MAP)
    from Object_detection.custom_utils import get_detection_transforms, view
    from Object_detection.model import load_model
except Exception:
    from config import (LABEL_MAP)
    from custom_utils import get_detection_transforms, view
    from model import load_model


class Detection:
    def __init__(self, path, model_path):
        self.path = path
        self.model_path = model_path

    def detection(self):
        _model = load_model(self.model_path)
        transforms = get_detection_transforms()

        image = Image.open(self.path).convert("RGB")

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Apply the transforms to the new image
        transformed = transforms(image=image_np)

        # Extract the transformed image tensor and bounding box labels
        image_tensor = torch.stack([transformed["image"].float() / 250.0])
        # labels = transformed["label"]

        # Run the object detection model on the new image
        _model.eval()
        with torch.no_grad():
            outputs = _model(image_tensor)

        print(outputs)

        # Extract the predicted labels and bounding boxes from the model output
        predicted_labels = outputs[0]['labels']
        predicted_boxes = outputs[0]['boxes']
        predicted_scores = outputs[0]['scores']

        nms_boxes = ops.nms(predicted_boxes, predicted_scores, iou_threshold=0.5)

        mask = (predicted_scores[nms_boxes] > 0.5)
        indices = np.where(mask)[0]

        predicted_labels_filtered = predicted_labels[indices]
        predicted_box_filtered = predicted_boxes[indices]
        predicted_scores_filtered = predicted_scores[indices]

        labels = [{'boxes': predicted_box_filtered, 'labels': predicted_labels_filtered}]

        key = None
        for k, v in LABEL_MAP.items():
            if v == predicted_labels_filtered[0]:
                key = k
                break

        print(f'Your flower is : {predicted_scores_filtered[0].item() * 100}%  {key}')

        # # Display the new image and predicted labels
        view(image_tensor, labels, 1, label_map=LABEL_MAP)


if __name__ == '__main__':
    image_path = 'F:/Project/MachineLearning/RMIT/Asm/Asm2/Final/data/Flowers/Rosy/rosy_150.jpg'
    object_detection_model_path = './output/model/final_model.pt'

    object_detection = Detection(image_path, object_detection_model_path)

    object_detection.detection()
