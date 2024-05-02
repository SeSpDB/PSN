#!/bin/bash
# conda init # 为了使用conda命令
conda activate /home/deeplearning/nas-files/conda/anaconda3/envs/tracer # 激活conda环境
# 设置Python脚本的路径
PYTHON_SCRIPT1="/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Step3_get_corpus_many.py"
PYTHON_SCRIPT2="/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Step4_shuffle_data.py"

# 先执行删除数据的命令
python3 $PYTHON_SCRIPT1 full 6
python3 $PYTHON_SCRIPT1 full 7
python3 $PYTHON_SCRIPT1 patch 6
python3 $PYTHON_SCRIPT1 patch 7
python3 $PYTHON_SCRIPT1 slice 6
python3 $PYTHON_SCRIPT1 slice 7
# full 
for i in {1..4}
do
    python3 $PYTHON_SCRIPT1 full "$i" >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_full.log 2>&1
    # 暂停5秒
    sleep 5
done
python3 $PYTHON_SCRIPT2 full >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_full.log 2>&1


sleep 500

# patch
for i in {1..4}
do
    python3 $PYTHON_SCRIPT1 patch "$i" >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_patch.log 2>&1
    # 暂停5秒
    sleep 5
done
python3 $PYTHON_SCRIPT2 patch >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_patch.log 2>&1

# slice 

sleep 500

for i in {1..4}
do
    python3 $PYTHON_SCRIPT1 slice  "$i" >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_slice.log 2>&1
    # 暂停5秒
    sleep 5
done
python3 $PYTHON_SCRIPT2 slice  >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/make_data_slice.log 2>&1