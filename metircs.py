import torch
import logging
import numpy as np
from typing import *
from allennlp.training.metrics import Metric

logger = logging.getLogger(__name__)



@Metric.register('multilbale-accuracy')
class AccuracyMultiLabel(Metric):
    def __init__(self, threshold=0.5):
        super(AccuracyMultiLabel, self).__init__()
        self._correct_count = 0.
        self._total_count = 0.
        self.threshold = threshold

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size()[-1]
        if gold_labels.size() != predictions.size():
            raise ValueError(f"gold_labels must have shape == predictions.size() but "
                             f"found tensor of shape: {gold_labels.size()}")
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(f"mask must have shape == predictions.size() but "
                             f"found tensor of shape: {mask.size()}")

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0].float()
        else:
            keep = torch.ones(batch_size).float()
        predictions = predictions.sigmoid()
        predictions = predictions.view(batch_size, -1).numpy()
        gold_labels = gold_labels.view(batch_size, -1).numpy()
        predictions = (predictions > self.threshold)
        keep = keep.numpy()
        # keep = np.reshape(keep, (np.shape(keep)[0], 1))
        correct = np.zeros_like(keep)
        for idx, (p, g) in enumerate(zip(predictions, gold_labels)):
            flag = int((p == g).all())
            correct[idx] = flag
            if flag:
                print(np.argwhere(p>0), np.argwhere(g>0))
        # correct = np.sum([]) (predictions == gold_labels)
        self._correct_count += np.sum((correct * keep))
        self._total_count += np.sum(keep)
        logging.debug(msg='multilabel accuracy: corrcet: {}, total: {}'.format(self._correct_count,
                                                                               self._total_count))

    def get_metric(self, reset: bool):
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
