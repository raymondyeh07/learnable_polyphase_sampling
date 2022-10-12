import numpy as np
from torchmetrics import ConfusionMatrix


class SegmentationMetrics(ConfusionMatrix):
    def compute(self):
        cfm = super().compute()
        return self.get_results(cfm.cpu().detach().numpy())

    def get_results(self, hist):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return {
            "Acc": acc,
            "Mean_Acc": acc_cls,
            "FreqW_Acc": fwavacc,
            "Mean_IoU": mean_iu,
            "Class_IoU": cls_iu,
        }
