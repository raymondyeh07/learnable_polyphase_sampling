import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
import pytorch_lightning as pl
from torch.optim import Optimizer, Adam
from .core import AbstractBaseClassifierModel
from layers.polydown import set_pool


class BasicClassifier( AbstractBaseClassifierModel):
    def __init__( self, input_shape, num_classes,
                  padding_mode= 'zeros', learning_rate= 0.1, pooling_layer= nn.AvgPool2d,
                  extras_model=None, **kwargs):
        super().__init__(**kwargs)
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, padding_mode=padding_mode)

        # Set pool
        self.pool1=set_pool(pooling_layer=pooling_layer,
                            p_ch=32,
                            h_ch=32)
        self.pool2=set_pool(pooling_layer=pooling_layer,
                            p_ch=64,
                            h_ch=64)

        # Set head
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64, num_classes)

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

