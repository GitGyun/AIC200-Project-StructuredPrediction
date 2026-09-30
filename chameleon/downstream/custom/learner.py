"""A Learner template for your own structured-prediction task.

A ``Learner`` is the task-specific half of Chameleon. The model produces raw logits; the learner
decides what those logits *mean*:

===========================  ==================================================================
method                       responsibility
===========================  ==================================================================
``compute_loss``             the training objective
``postprocess_logits``       logits -> a ``[0, 1]`` map, still at model resolution
``postprocess_final``        the map -> the final prediction (e.g. thresholded mask)
``postprocess_vis``          the map -> an RGB image a human can look at
``compute_metric``           the evaluation number
===========================  ==================================================================

``CustomLearner`` implements all of them for three common label types, selected with
``config.loss_type``:

* ``bce``  -- binary / multi-label masks. Metric: mean IoU.
* ``l2``   -- continuous maps (depth, density, ...). Metric: mean absolute error.
* ``ssl``  -- keypoint heatmaps (spatial softmax). Metric: mean absolute error of the argmax.

For anything else, subclass this and override the methods you need; the built-in learners in
``downstream/{ap10k,isic2018,fsc147,cellpose,davis2017}/learner.py`` are worked examples.
"""

import torch
import torch.nn.functional as F

from train.loss import spatial_softmax_loss
from train.visualize import postprocess_semseg

from ..base_learner import BaseLearner
from .dataset import CustomDenseDataset


class CustomLearner(BaseLearner):
    BaseDataset = CustomDenseDataset

    # ------------------------------------------------------------------ setup

    def register_evaluator(self):
        keys = ['mtest_train', 'mtest_valid'] if self.config.stage == 1 else ['mtest_test']
        self.evaluator = {key: [] for key in keys}

    def reset_evaluator(self):
        for key in self.evaluator:
            self.evaluator[key] = []

    # ------------------------------------------------------------------- loss

    def compute_loss(self, Y_pred, Y, M):
        loss_type = getattr(self.config, 'loss_type', 'bce')
        if loss_type == 'bce':
            loss = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
        elif loss_type == 'l2':
            loss = (M * F.mse_loss(Y_pred.sigmoid(), Y, reduction='none')).mean()
        elif loss_type == 'ssl':
            loss = spatial_softmax_loss(Y_pred, Y, M, reduction='mean',
                                        scaled=getattr(self.config, 'scale_ssl', False))
        else:
            raise NotImplementedError(
                f'loss_type "{loss_type}" is not handled by CustomLearner. '
                f'Use one of bce / l2 / ssl, or override compute_loss().'
            )
        return loss, {'loss': loss.detach(), f'loss_{loss_type}': loss.detach()}

    # --------------------------------------------------------- post-processing

    def postprocess_logits(self, Y_pred_out):
        loss_type = getattr(self.config, 'loss_type', 'bce')
        if loss_type == 'ssl':
            # spatial softmax: each channel becomes a probability map over pixels, peak-normalised
            from einops import rearrange, reduce
            H, W = Y_pred_out.shape[-2:]
            flat = rearrange(Y_pred_out, '1 T N C H W -> 1 T N C (H W)').softmax(dim=-1)
            Y_pred_out = rearrange(flat, '1 T N C (H W) -> 1 T N C H W', H=H, W=W)
            Y_pred_out = Y_pred_out / (1e-18 + reduce(Y_pred_out, '1 T N C H W -> 1 T N C 1 1', 'max'))
            return Y_pred_out
        return Y_pred_out.sigmoid()

    def postprocess_final(self, Y_pred):
        if getattr(self.config, 'loss_type', 'bce') == 'bce':
            return (Y_pred > 0.5).to(Y_pred.dtype)
        return Y_pred

    def postprocess_vis(self, label, img=None, aux=None):
        loss_type = getattr(self.config, 'loss_type', 'bce')
        if loss_type == 'bce':
            return postprocess_semseg(label, img, aux)
        # continuous / heatmap labels: show the raw map, normalised per image
        if label.ndim == 3:
            label = label.unsqueeze(1)
        label = label.float()
        if label.shape[1] > 3:
            label = label.max(dim=1, keepdim=True).values  # collapse many channels into one map
        peak = label.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        return (label / peak).clip(0, 1)

    # ----------------------------------------------------------------- metrics

    def compute_metric(self, Y, Y_pred, M, aux, evaluator_key=None):
        loss_type = getattr(self.config, 'loss_type', 'bce')
        if Y_pred.ndim == 3:
            Y_pred = Y_pred.unsqueeze(1)
        Y, Y_pred, M = Y.float(), Y_pred.float(), M.float()

        if loss_type == 'bce':
            pred = (Y_pred > 0.5).float()
            target = (Y > 0.5).float()
            inter = (pred * target * M).sum((1, 2, 3))
            union = (((pred + target) > 0).float() * M).sum((1, 2, 3))
            metric = (inter / union.clamp(min=1e-8)).mean()
        else:
            metric = ((Y_pred - Y).abs() * M).sum() / M.sum().clamp(min=1e-8)

        if evaluator_key is not None and evaluator_key in self.evaluator:
            self.evaluator[evaluator_key].append(metric.detach().cpu())
        return metric

    def log_metrics(self, loss_pred, log_dict, valid_tag):
        name = 'IoU' if getattr(self.config, 'loss_type', 'bce') == 'bce' else 'MAE'
        # the training loop minimises this, so IoU is logged inverted
        value = 1 - loss_pred if name == 'IoU' else loss_pred
        log_dict[f'{valid_tag}/{self.vis_tag}_{name}{"_inverted" if name == "IoU" else ""}'] = value

    def get_test_metrics(self, metrics_total):
        name = 'IoU' if getattr(self.config, 'loss_type', 'bce') == 'bce' else 'MAE'
        return [(name, float(metrics_total))]
