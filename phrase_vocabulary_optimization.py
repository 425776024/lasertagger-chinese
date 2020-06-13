# coding=utf-8


from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Sequence, Text

from absl import app
from absl import flags
from absl import logging

from src.utils import utils

import numpy as np
import scipy.sparse
import tensorflow as tf
from src.compute_lcs import _compute_lcs
from src.curLine_file import curLine

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', '/Users/jiang/Documents/Github/text_scalpel/corpus/rephrase_corpus/train.txt',
    'Path to the input file containing source-target pairs from which the '
    'vocabulary is optimized (see `input_format` flag and utils.py for '
    'documentation).')
flags.DEFINE_enum(
    'input_format', 'wikisplit', ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the `input_file`. See utils.py for '
    'documentation on the different formats.')
flags.DEFINE_integer(
    'max_input_examples', 50000,
    'At most this many examples from the `input_file` are used for optimizing '
    'the vocabulary.')
flags.DEFINE_string(
    'output_file', '/Users/jiang/Documents/Github/text_scalpel/output/label_map.txt',
    'Path to the resulting file with all possible tags. Coverage numbers will '
    'be written to a separate file which has the same path but ".log" appended '
    'to it.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.开不开启置换')
flags.DEFINE_integer('vocabulary_size', 500,
                     'Number of phrases to include in the vocabulary.')
flags.DEFINE_integer(
    'num_extra_statistics', 100,
    'Number of extra phrases that are not included in the vocabulary but for '
    'which we compute the coverage numbers. These numbers help determining '
    'whether the vocabulary size should have been larger.')


def _get_added_phrases(source: Text, target: Text) -> Sequence[Text]:
    """
    Computes the phrases that need to be added to the source to get the target.
    计算需要被加入进去的短句，且尽可能长
    英文是分成word sep=' '，中文是分成 字 sep=''
    """
    sep = ''
    source_tokens = utils.get_token_list(source.lower())
    target_tokens = utils.get_token_list(target.lower())
    # s1={1,3,4,5,6,7,7,8},s2={3,5,7,4,8,6,7,8,2} return 35778
    kept_tokens = _compute_lcs(source_tokens, target_tokens)
    # 找到这对数据里面，有多少个不同的（尽可能长）短句，是被活活加进去的
    added_phrases = []
    kept_idx = 0
    phrase = []
    for token in target_tokens:
        # 尽可能长找到 被加入进去的短句
        if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
            kept_idx += 1
            if phrase:
                added_phrases.append(sep.join(phrase))
                phrase = []
        else:
            phrase.append(token)
    if phrase:
        added_phrases.append(sep.join(phrase))
    return added_phrases


def _added_token_counts(data_iterator, try_swapping, max_input_examples=10000):
    """
    Computes how many times different phrases have to be added.
    计算不同的短句，被添加多少次，以这些来建立优化的词汇表
    Args:
      data_iterator: 迭代器，yield source lists and targets.
        See function yield_sources_and_targets in utils.py
      try_swapping: 是否开启交换.
      max_input_examples: Maximum number of examples to be read from the iterator.
    """
    phrase_counter = collections.Counter()
    num_examples = 0
    all_added_phrases = []
    max_seq_length = 0
    for sources, target in data_iterator:
        # sources 可能是多句话，后面用空格拼接起来
        if num_examples >= max_input_examples:
            break
        source_merge = ' '.join(sources)
        if len(source_merge) > max_seq_length:
            print(curLine(), "max_seq_length=%d, len(source_merge)=%d,source_merge:%s" %
                  (max_seq_length, len(source_merge), source_merge))
            max_seq_length = len(source_merge)
        logging.log_every_n(logging.INFO, f'{num_examples} examples processed.', 10000)
        added_phrases = _get_added_phrases(source_merge, target)
        if try_swapping and len(sources) == 2:
            added_phrases_swap = _get_added_phrases(' '.join(sources[::-1]), target)
            # 稍微"swap交换"一丢丢(长度更小)，就能达到和你"add添加"一样的内容，那肯定选择交换操作
            if len(''.join(added_phrases_swap)) < len(''.join(added_phrases)):
                added_phrases = added_phrases_swap
        # 统计短句的次数
        for phrase in added_phrases:
            phrase_counter[phrase] += 1
        all_added_phrases.append(added_phrases)
        num_examples += 1
    logging.info(f'{num_examples} examples processed.\n')
    # 短句Counter，全部要add的短句,最大长度
    return phrase_counter, all_added_phrases, max_seq_length


def _construct_added_phrases_matrix(all_added_phrases, phrase_counter):
    """Constructs a sparse phrase occurrence matrix.

    Examples are on rows and phrases on columns.

    Args:
      all_added_phrases: List of lists of added phrases (one list per example).
      phrase_counter: Frequence of each unique added phrase.

    Returns:
      Sparse boolean matrix whose element (i, j) indicates whether example i
      contains the added phrase j. Columns start from the most frequent phrase.
    """
    phrase_2_idx = {
        tup[0]: i for i, tup in enumerate(phrase_counter.most_common())
    }
    matrix = scipy.sparse.dok_matrix((len(all_added_phrases), len(phrase_2_idx)),
                                     dtype=np.bool)
    for i, added_phrases in enumerate(all_added_phrases):
        for phrase in added_phrases:
            phrase_idx = phrase_2_idx[phrase]
            matrix[i, phrase_idx] = True
    # Convert to CSC format to support more efficient column slicing.
    return matrix.tocsc()


def _count_covered_examples(matrix, vocabulary_size):
    """Returns the number of examples whose added phrases are in the vocabulary.

    This assumes the vocabulary is created simply by selecting the
    `vocabulary_size` most frequent phrases.

    Args:
      matrix: Phrase occurrence matrix with the most frequent phrases on the
        left-most columns.
      vocabulary_size: Number of most frequent phrases to include in the
        vocabulary.
    """
    # Ignore the `vocabulary_size` most frequent (i.e. leftmost) phrases (i.e.
    # columns) and count the rows with zero added phrases.
    return (matrix[:, vocabulary_size:].sum(axis=1) == 0).sum()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')

    data_iterator = utils.yield_sources_and_targets(FLAGS.input_file,
                                                    FLAGS.input_format)
    phrase_counter, all_added_phrases, max_seq_length = _added_token_counts(
        data_iterator, FLAGS.enable_swap_tag, FLAGS.max_input_examples)
    matrix = _construct_added_phrases_matrix(all_added_phrases, phrase_counter)
    num_examples = len(all_added_phrases)

    statistics_file = FLAGS.output_file + '.log'
    with tf.io.gfile.GFile(FLAGS.output_file, 'w') as writer:
        with tf.io.gfile.GFile(statistics_file, 'w') as stats_writer:
            stats_writer.write('Idx\tFrequency\tCoverage (%)\tPhrase\n')
            writer.write('KEEP\n')
            writer.write('DELETE\n')
            if FLAGS.enable_swap_tag:
                writer.write('SWAP\n')
            for i, (phrase, count) in enumerate(
                    phrase_counter.most_common(FLAGS.vocabulary_size +
                                               FLAGS.num_extra_statistics)):
                # Write tags.
                if i < FLAGS.vocabulary_size:
                    writer.write(f'KEEP|{phrase}\n')
                    writer.write(f'DELETE|{phrase}\n')
                # Write statistics.
                # 用前ｉ＋１个高频ｐｈｒａｓｅ能覆盖的语料的比例
                coverage = 100.0 * _count_covered_examples(matrix, i + 1) / num_examples
                stats_writer.write(f'{i + 1}\t{count}\t{coverage:.2f}\t{phrase}\n')
    logging.info(f'Wrote tags to: {FLAGS.output_file}')
    logging.info(f'Wrote coverage numbers to: {statistics_file}')
    print(curLine(), "max_seq_length=", max_seq_length)


if __name__ == '__main__':
    app.run(main)
