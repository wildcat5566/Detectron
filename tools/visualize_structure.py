from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

def get_model(cfg_file, weights_file):
    merge_cfg_from_file(cfg_file)
    cfg.TRAIN.WEIGHTS = '' # NOTE: do not download pretrained model weights
    cfg.TEST.WEIGHTS = weights_file
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    return model

cfg_file = '{}/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'.format(DETECTRON_ROOT)
weights_file = 'https://s3-us-west-2.amazonaws.com/detectron/' + \
    '35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/' + \
    'output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
model = get_model(cfg_file, weights_file)

from caffe2.python import net_drawer
g = net_drawer.GetPydotGraph(model, rankdir="TB")
g.write_dot(model.Proto().name + '.dot')
