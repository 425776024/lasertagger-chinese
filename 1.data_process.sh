# 扩充文本匹配的语料  文本复述任务
# python -m pip install --upgrade pip -i https://pypi.douban.com/simple
# pip install -i https://pypi.douban.com/simple/ bert-tensorflow==1.0.1
# pip install -i https://pypi.douban.com/simple/ tensorflow==1.15.0

# 最傻的办法在shell中运行conda指定环境
export python=/home/jiang/anaconda3/envs/tf15_py37/bin/python3
# 然后：${python} xxx.py

# set gpu id to use
export CUDA_VISIBLE_DEVICES=""

start_tm=$(date +%s%N)

export Root_Dir="${HOME}/Documents/Github/text_scalpel"

# 训练多少个epoch
export num_train_epochs=3
# 训练的BATCH_SIZE
export TRAIN_BATCH_SIZE=256
# 训练的词汇表大小，（还会经过优化，不一定是这么多）
export PHRASE_VOCAB_SIZE=500
# 最大输入，训练数量
export MAX_INPUT_EXAMPLES=100000
# 多少step保存一次模型
export SAVE_CHECKPOINT_STEPS=200
# 是否开启4种编辑中的：SWAP（交换）编辑
export enable_swap_tag=false
#是否引入额外词汇表
export output_arbitrary_targets_for_infeasible_examples=false
#工作语料目录
export DATA_DIR="${Root_Dir}/corpus/rephrase_corpus"
#BERT目标，这里是轻量级的RoBERTa-tiny-clue，如果用别的bert，configs/lasertagger_config.json 也要改改
export BERT_BASE_DIR="${Root_Dir}/bert_base/RoBERTa-tiny-clue"
#输出目录
export OUTPUT_DIR="${Root_Dir}/output"

echo 'run:phrase_vocabulary_optimization.py,开始建立，优化词汇表'
${python} phrase_vocabulary_optimization.py \
  --input_file=${DATA_DIR}/train.txt \
  --input_format=wikisplit \
  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
  --max_input_examples=${MAX_INPUT_EXAMPLES} \
  --enable_swap_tag=${enable_swap_tag} \
  --output_file=${OUTPUT_DIR}/label_map.txt

export max_seq_length=40

echo 'run:preprocess_main.py,开始整理数据'
${python} preprocess_main.py \
  --input_file=${DATA_DIR}/train.txt \
  --input_format=wikisplit \
  --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}

${python} preprocess_main.py \
  --input_file=${DATA_DIR}/tune.txt \
  --input_format=wikisplit \
  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}
