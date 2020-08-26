import cv2
import numpy as np
from tensorflow.python.keras import backend as K

def mean_iou(num_classes):
    """Returns a Intersection over Union (IoU) metric metrics function. 
       The function returns the mean IOU over all classes.
       Args:
           num_classes: scalar with number of classes
       Return: metric function
    """
    def mean_iou(y_true, y_pred):
        """
        Args:
           y_true: true labels, tensor with shape (-1, num_labels)
           y_pred: predicted label propabilities from a softmax layer,
                tensor with shape (-1, num_labels, num_classes)
        """
        iou_sum = K.variable(0.0, name='iou_sum')
        seen_classes = K.variable(0.0, name='seen_classes')
        y_true_sparse = K.argmax(y_true, axis=-1)
        y_pred_sparse = K.argmax(y_pred, axis=-1)

        for c in range(0, num_classes):
            true_c = K.cast(K.equal(y_true_sparse, c), K.floatx())
            pred_c = K.cast(K.equal(y_pred_sparse, c), K.floatx())

            true_c_sum = K.sum(true_c)
            pred_c_sum = K.sum(pred_c)

            intersect = true_c * pred_c
            union = true_c + pred_c - intersect

            intersect_sum = K.sum(intersect)
            union_sum = K.sum(union)

            iou = intersect_sum / union_sum
            union_sum_is_zero = K.equal(union_sum, 0)

            iou_sum = K.switch(union_sum_is_zero,
                        iou_sum,
                        iou_sum+iou)

            seen_classes = K.switch(union_sum_is_zero,
                                    seen_classes,
                                    seen_classes+1)

        # Calculate mean IOU over all (seen) classes. Regarding this check
        # `seen_classes` can only be 0 if none of the true or predicted 
        # labels in the batch contains a valid class. We do not want to
        # raise a DivByZero error in this case.
        return K.switch(K.equal(seen_classes, 0),
                                iou_sum,
                                iou_sum / seen_classes)

    return mean_iou

def onehot_mean_iou(num_classes):
    # TODO: check this, output is [0, 1] and lb is 0 or 1 
    """Returns a Intersection over Union (IoU) metric metrics function. 
       The function returns the mean IOU over all classes.
       Args:
           num_classes: scalar with number of classes
       Return: metric function
    """
    def mean_iou(y_true, y_pred):
        """
        Args:
           y_true: true labels, tensor with shape (-1, num_labels)
           y_pred: predicted label propabilities from a softmax layer,
                tensor with shape (-1, num_labels, num_classes)
        """
        iou_sum = K.variable(0.0, name='iou_sum')
        seen_classes = K.variable(0.0, name='seen_classes')
        # y_pred_sparse = K.argmax(y_pred, axis=-1)

        for c in range(0, num_classes):
            # true_c = K.cast(K.equal(y_true[..., c], c), K.floatx())
            true_c = K.cast(y_true[..., c], K.floatx())
            pred_c = K.cast(y_pred[..., c], K.floatx())

            true_c_sum = K.sum(true_c)
            pred_c_sum = K.sum(pred_c)

            intersect = true_c * pred_c
            union = true_c + pred_c - intersect

            intersect_sum = K.sum(intersect)
            union_sum = K.sum(union)

            iou = intersect_sum / union_sum
            union_sum_is_zero = K.equal(union_sum, 0)

            iou_sum = K.switch(union_sum_is_zero,
                        iou_sum,
                        iou_sum+iou)

            seen_classes = K.switch(union_sum_is_zero,
                                    seen_classes,
                                    seen_classes+1)

        # Calculate mean IOU over all (seen) classes. Regarding this check
        # `seen_classes` can only be 0 if none of the true or predicted 
        # labels in the batch contains a valid class. We do not want to
        # raise a DivByZero error in this case.
        return K.switch(K.equal(seen_classes, 0),
                                iou_sum,
                                iou_sum / seen_classes)

    return mean_iou