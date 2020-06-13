# 最傻的办法在shell中运行conda指定环境
export python=/home/jiang/anaconda3/envs/tf15_py37/bin/python3
# 然后：${python} xxx.py

export CUDA_VISIBLE_DEVICES=""
start_tm=$(date +%s%N)

export Root_Dir="${HOME}/Documents/Github/text_scalpel"
export WIKISPLIT_DIR="${Root_Dir}/corpus/rephrase_corpus"
export OUTPUT_DIR="${Root_Dir}/output"

export EXPERIMENT=cefect
PREDICTION_FILE=${OUTPUT_DIR}/models/pred.tsv

#### 5. Evaluation
${python} score_main.py --prediction_file=${PREDICTION_FILE}

end_tm=$(date +%s%N)
use_tm=$(echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}')
echo "cost time" $use_tm "h"
