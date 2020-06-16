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

# 训练时，最多保存多少个step模型。3？  训练了10，20，30，40，50 step时，只保存最后的30,40,50，类似保存：栈大小
export keep_checkpoint_max=3
# 最多训练多少轮
export num_train_epochs=3
# How many steps to make in each estimator call
export iterations_per_loop=1000
# 训练的BATCH_SIZE
export TRAIN_BATCH_SIZE=256
# 多少step保存一次模型
export SAVE_CHECKPOINT_STEPS=200
# 是否开启4种编辑中的：SWAP（交换）编辑
export enable_swap_tag=false
#是否引入额外词汇表
export output_arbitrary_targets_for_infeasible_examples=false
#工作语料目录
export Root_Dir="${HOME}/Documents/Github/text_scalpel"
export WIKISPLIT_DIR="${Root_Dir}/corpus/rephrase_corpus"
#BERT目标，这里是轻量级的RoBERTa-tiny-clue，如果用别的bert，configs/lasertagger_config.json 也要改改
export BERT_BASE_DIR="${Root_Dir}/bert_base/RoBERTa-tiny-clue"
#输出目录
export OUTPUT_DIR="${Root_Dir}/output"

# 句子大概最长多长
export max_seq_length=40

# 训练的数据量
export NUM_TRAIN_EXAMPLES=67739
# eval数据量
export NUM_EVAL_EXAMPLES=804
export CONFIG_FILE="${Root_Dir}/configs/lasertagger_config.json"

${python} run_lasertagger.py \
  --training_file=${OUTPUT_DIR}/train.tf_record \
  --eval_file=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/ \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --do_train=true \
  --keep_checkpoint_max=${keep_checkpoint_max} \
  --iterations_per_loop=${iterations_per_loop} \
  --do_eval=false \
  --num_train_epochs=${num_train_epochs} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
  --max_seq_length=${max_seq_length} \
  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
  --num_eval_examples=${NUM_EVAL_EXAMPLES}
