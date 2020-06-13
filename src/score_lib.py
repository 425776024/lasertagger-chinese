# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions for computing evaluation metrics."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import re

from src import sari_hook
from src.utils import utils

import tensorflow as tf


def read_data(
    path,
    lowercase):
  """Reads data from prediction TSV file.

  The prediction file should contain 3 or more columns:
  1: sources (concatenated)
  2: prediction
  3-n: targets (1 or more)

  Args:
    path: Path to the prediction file.
    lowercase: Whether to lowercase the data (to compute case insensitive
      scores).

  Returns:
    Tuple (list of sources, list of predictions, list of target lists)
  """
  sources = []
  predictions = []
  target_lists = []
  with tf.io.gfile.GFile(path) as f:
    for line in f:
      source, pred, *targets = line.rstrip('\n').split('\t')
      if lowercase:
        source = source.lower()
        pred = pred.lower()
        targets = [t.lower() for t in targets]
      sources.append(source)
      predictions.append(pred)
      target_lists.append(targets)
  return sources, predictions, target_lists


def compute_exact_score(predictions,
                        target_lists):
  """Computes the Exact score (accuracy) of the predictions.

  Exact score is defined as the percentage of predictions that match at least
  one of the targets.

  Args:
    predictions: List of predictions.
    target_lists: List of targets (1 or more per prediction).

  Returns:
    Exact score between [0, 1].
  """
  num_matches = sum(any(pred == target for target in targets)
                    for pred, targets in zip(predictions, target_lists))
  # num_matches=0.0
  # for pred, targets in zip(predictions, target_lists):
  #   for target in targets:
  #     if pred == target:
  #       num_matches += 1
  #       break
  return num_matches / max(len(predictions), 0.1)  # Avoids 0/0.


def compute_sari_scores(
    sources,
    predictions,
    target_lists,
    ignore_wikisplit_separators = True
):
  """Computes SARI scores.

  Wraps the t2t implementation of SARI computation.

  Args:
    sources: List of sources.
    predictions: List of predictions.
    target_lists: List of targets (1 or more per prediction).
    ignore_wikisplit_separators: Whether to ignore "<::::>" tokens, used as
      sentence separators in Wikisplit, when evaluating. For the numbers
      reported in the paper, we accidentally ignored those tokens. Ignoring them
      does not affect the Exact score (since there's usually always a period
      before the separator to indicate sentence break), but it decreases the
      SARI score (since the Addition score goes down as the model doesn't get
      points for correctly adding <::::> anymore).

  Returns:
    Tuple (SARI score, keep score, addition score, deletion score).
  """
  sari_sum = 0
  keep_sum = 0
  add_sum = 0
  del_sum = 0
  for source, pred, targets in zip(sources, predictions, target_lists):
    if ignore_wikisplit_separators:
      source = re.sub(' <::::> ', ' ', source)
      pred = re.sub(' <::::> ', ' ', pred)
      targets = [re.sub(' <::::> ', ' ', t) for t in targets]
    source_ids = utils.get_token_list(source)
    pred_ids = utils.get_token_list(pred)
    list_of_targets = [utils.get_token_list(t) for t in targets]
    sari, keep, addition, deletion = sari_hook.get_sari_score(
        source_ids, pred_ids, list_of_targets, beta_for_deletion=1)
    sari_sum += sari
    keep_sum += keep
    add_sum += addition
    del_sum += deletion
  n = max(len(sources), 0.1)  # Avoids 0/0.
  return (sari_sum / n, keep_sum / n, add_sum / n, del_sum / n)
