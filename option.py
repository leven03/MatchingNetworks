##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Matching Network')
        parser.add_argument('--dataroot', type=str, default='omniglot_dataset', help='path to dataset')
        parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
        parser.add_argument('--classes_per_set', type=int, default=5, help='Number of classes to sample per set')
        parser.add_argument('--samples_per_class', type=int, default=5, help='Number of samples per set to sample')
        parser.add_argument('--log-dir', default='./logs', help='folder to output model checkpoints')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
