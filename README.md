# LaserTagger-Chinese

- 用中文学习、训练、验证[谷歌的LaserTagger模型](https://github.com/google-research/lasertagger)
- 参考于这个博主的[中文案例](https://github.com/Mleader2/text_scalpel) 进行些小改&训练

---

# 主要工作
- 对原来的代码``（结构）``进行了些``整理``（原来的是真乱）
- 添加了些``注释``（我真看了它源码）
- 更``舒服的shell``运行案例（开箱即学）
- 进行了自己的数据的训练，``效果确实不错``【可摘要、可文本复述、训练快、推断快】
- 给出``案例数据``，方便学习
- bert模型（RoBERTa-tiny-clue）直接上传了，也不大，直接下载

# 运行方式
- 记得准备corpus/rephrase_corpus 那种数据（test.txt、train.txt、tune.txt）
- 记得安装包：pip install requirements.txt
- 其中的`export python=/home/xxx/anaconda3/envs/tf15_py37/bin/python3`记得改成你自己的python环境路径
- 其中的`export Root_Dir=xx`记得改成你自己的代码根目录
```shell script
# shell里面给出了参数注释
# 处理数据
sh 1.data_process.sh
# 训练
sh 2.train.sh
# 导出pb模型
sh 3.export.sh
# 预测
sh 4.predict.sh
# 计算预测后的分数
sh 5.eval_score.sh
```

# PS 

- 用的这个轻量bert：RoBERTa-tiny-clue，很小很快，效果也不差
>bert数据（RoBERTa-tiny-clue）直接放进来了，在bert_base/RoBERTa-tiny-clue
>所以你不用去别的地方下载了
- 如果想换成base bert，参数改成
>和[谷歌的LaserTagger模型中configs/lasertagger_config.json](https://github.com/google-research/lasertagger/blob/master/configs/lasertagger_config.json)
中那样的，因为RoBERTa-tiny-clue参数要小很多

- 案例数据，在[corpus/rephrase_corpus](corpus/)下




