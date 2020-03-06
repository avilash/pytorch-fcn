import numpy as np


def mIoU(predictions, labels, num_classes):
    ious = []
    for class_idx in range(num_classes):
        class_labels = labels == class_idx
        class_preds = predictions == class_idx
        intersection = np.logical_and(class_labels, class_preds)
        union = np.logical_or(class_labels, class_preds)
        class_iou = np.sum(intersection) / np.sum(union)
        ious.append(class_iou)
    return np.mean(ious)
