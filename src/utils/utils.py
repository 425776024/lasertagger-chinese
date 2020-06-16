# coding=utf-8

# Lint as: python3
"""Utility functions for LaserTagger."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
from typing import Iterator, Mapping, Sequence, Text, Tuple

import tensorflow as tf


def get_token_list(text):
    # 你厉害了！：[你,厉,害,了,！]
    return list(text)


def _calculate_steps(num_examples, batch_size, num_epochs, warmup_proportion=0):
    """Calculates the number of steps.
      warmup_proportion: Proportion of warmup steps.
    Returns:
      Tuple (number of steps, number of warmup steps).
    """
    steps = int(num_examples / batch_size * num_epochs)
    warmup_steps = int(warmup_proportion * steps)
    return steps, warmup_steps


def yield_sources_and_targets(
        input_file,
        input_format):
    """Reads and yields source lists and targets from the input file.

    Args:
      input_file: Path to the input file.
      input_format: Format of the input file.

    Yields:
      Tuple with (list of source texts, target text).
    """

    def _yield_wikisplit_examples(
            input_file):
        # 把数据（corpus/rephrase_corpus/xxx.txt），组成：(原始（可一句可多句话组成数组）,目标) 对，迭代出去
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                source, target = line.rstrip('\n').replace('\ufeff', '').split('\t')
                if len(source) <= 2 or len(target) <= 2:
                    continue
                yield [source], target

    for sources, target in _yield_wikisplit_examples(input_file):
        yield sources, target


def read_label_map(path):
    """Returns label map read from the given path."""
    with tf.io.gfile.GFile(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
            label_map = {}
            empty_line_encountered = False
            for tag in f:
                tag = tag.strip()
                if tag:
                    label_map[tag] = len(label_map)
                else:
                    if empty_line_encountered:
                        raise ValueError(
                            'There should be no empty lines in the middle of the label map '
                            'file.'
                        )
                    empty_line_encountered = True
            return label_map
