#! /usr/bin/env python
#coding=utf-8
import os,sys
import random

def load_and_shuffle_data(directory):
    data = []  # 用于存储读取的数据
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # 确保只处理.txt文件
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                data.extend(lines)  # 添加文件内容到数据列表

    # 打乱数据
    random.shuffle(data)
    
    return data

def save_shuffled_data(data, output_dir, lines_per_file=1000,task_name = ""):
    random.shuffle(data) # 打乱数据
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录，如果不存在
    
    file_count = 0
    for i in range(0, len(data), lines_per_file):
        file_count += 1
        with open(os.path.join(output_dir, task_name + "_shuffled_data_"+str(file_count)+".txt"), 'w') as file:
            file.writelines(data[i:i+lines_per_file])

# 使用示例
task = sys.argv[1] # full / patch / slice

if task == "full":
    # groups = ['slice1215', 'slice20231128', 'slice20240119',"full"]
    groups = ["full"]

elif task == "patch":
    # groups = ['slice1215_patch', 'slice20231128_patch', 'slice20240119_patch',"patch"]
    groups = ["patch"]

else:
    # groups = ['slice1215_slicing', 'slice20231128_slicing', 'slice20240119_slicing',"slice"]
    groups = ["slice"]

SAVE_PAIR_Path = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/"
shuffled_data = []
for data_file in groups:
    directory = SAVE_PAIR_Path + data_file  # 设置你的文件目录
    print("processing ",directory,task)
    output_directory = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/shuf_data/" + task  # 设置输出文件夹的路径
    shuffled_data = shuffled_data + load_and_shuffle_data(directory)
save_shuffled_data(shuffled_data, output_directory, lines_per_file=10000,task_name = task)  # 每个文件1000行

print("Data has been shuffled and saved in"+ output_directory )

# slice1215  slice1215_patch slice1215_slicing  slice20231128   slice20231128_patch   slice20231128_slicing   slice20240119 slice20240119_patch slice20240119_slicing