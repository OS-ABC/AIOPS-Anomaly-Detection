#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import config
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


options = config.parse_args()
seed_everything(seed=1234)


def train():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_supervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
