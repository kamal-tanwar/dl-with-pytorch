import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys_argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of workers for background data loading',
                            default=8,
                            type=int
                            )
        
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_mps = torch.backends.mps.is_available()
        self.device = torch.device('mps' if self.use_mps else 'cpu')
    
    def initModel(self):
        model = LunaModel()
        if self.use_mps:
            log.info('Using MPS; {} devices'.format(torch.backends.mps.device_count()))
        if torch.backends.mps.device_count() > 0:
            model = nn.DataParallel(model)
        model = model.to(device=self.device)

        return model
    
    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride = 10,
            isvalSet_bool = False
        )

        batch_size = self.cli_args.batch_size
        if self.use_mps:
            batch_size *= torch.backends.mps.device_count()
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_mps
        )

        return train_dl
    
    def initValDl(self):
        val_ds = LunaDataset(
            val_stride = 10,
            isvalSet_bool = True
        )

        batch_size = self.cli_args.batch_size
        if self.use_mps:
            batch_size *= torch.backends.mps.device_count()
        
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_mps
        )

        return val_dl
    
    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        








if __name__ == "__main__":
    LunaTrainingApp.main()