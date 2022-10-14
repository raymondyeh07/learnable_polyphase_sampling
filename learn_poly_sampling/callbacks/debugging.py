import pytorch_lightning as pl
from pytorch_lightning import Callback


class PrintBufferCallback(pl.Callback):
    def __init__(self, filter_name, **kwargs):
        self.filter_name = filter_name
        super().__init__(**kwargs)

    def on_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for n, p in pl_module.named_buffers():
            if self.filter_name in n:
                print(n, p)

        for n, p in pl_module.named_parameters():
            if self.filter_name in n:
                print(n, p)