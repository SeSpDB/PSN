# BEGIN: 提取列表
from curses import flash
from genericpath import isdir
from operator import imod
from re import T

from numpy import save, size, source
import pandas as pd
import threading
import os,sys
import shutil
import csv
from concurrent.futures import ThreadPoolExecutor
# from . config import TrainingConfig
# 修改为上三级目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import TrainingConfig
config = TrainingConfig()
from data_process import normalization
from data_process import patch_process
def extract_cve_commit_list(dataset_path):
    
    cve_commit_dict = {}
    
    for cve_list in os.listdir(dataset_path):
        
        data_path = dataset_path + "/" + cve_list

        commit_lists =  os.listdir(data_path)
        for commit_list in commit_lists:
            commit_path = data_path + commit_list
            if cve_list +","+ commit_list not in cve_commit_dict:
                cve_commit_dict[cve_list +","+ commit_list ] = []
            print(commit_path)
            if not os.path.isdir(commit_path):
                continue
            commitvar_list =  os.listdir(commit_path)
            for commit_var in commitvar_list:
                print("Processing: ", cve_list, commit_var)
                cve_commit_dict[cve_list +","+ commit_list].append(commit_var)
                
    return cve_commit_dict

def save_result_as_txt(cve_commit_dict, save_path):
    with open(save_path, 'w') as f:
        for cve,variants in cve_commit_dict.items():
            #f.write("[" + cve +","+commit + ': ' + ', '.join(variants) + '\n')
            f.write(cve  + ': ' + ', '.join(variants) + '\n')

def read_file_content(file_path):
    result_line = []
    with open(file_path, "r",encoding='iso-8859-1') as f:
        for line in f:
            temp_line = []
            temp_line.append(line)
            result_line.append(temp_line)
    return result_line

def process_file(source_path, target_path):
    """处理单个文件的复制和内容处理"""
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        patch_content = read_file_content(target_path)
        get_line, testfunc = normalization.mapping(patch_content)
        print("Copying: ", source_path, target_path)
        with open(target_path, 'w') as file:
            for line in get_line:
                file.write(str(line) + '\n')

def find_and_process_files(cve, commit, commit_variants, year, owner_path, target_dir):
    for owner_list in os.listdir(owner_path):
        branch_path = os.path.join(owner_path, owner_list)
        for branch_list in os.listdir(branch_path):
            commit_path = os.path.join(branch_path, branch_list)

            for commit_variants_list in commit_variants:
                var_folder_name = commit_variants_list.split('_')[0] + "_var"
                var_path = patch_process.find_variants(commit_path, var_folder_name)
                if var_path:
                    var_path = var_path[0]  # Take first match
                    source_path = os.path.join(var_path, commit_variants_list.split("_")[0] + "_" + commit_variants_list.split("_")[-1] + ".patch")
                    target_path = os.path.join(target_dir, cve, commit, commit_variants_list.split("_")[0] + "_" + commit_variants_list.split("_")[-1] + ".patch")
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(source_path, target_path)
                        print("Copying: ", source_path, target_path)
                        patch_content = read_file_content(target_path)
                        get_line, testfunc = normalization.mapping(patch_content)
                        with open(target_path, 'w') as file:
                            for line in get_line:
                                file.write(str(line) + '\n')
                    except Exception as e:
                        print(f"Error processing file {source_path}: {e}")

def copy_variant_files(source_dir="", target_dir="", copy_path="", num_workers=40):
    with open(copy_path, 'r') as file:
        reader = csv.reader(file, delimiter=':')
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for row in reader:
                cve_commit, commit_variants = row[0].strip(), row[1].strip()
                cve, commit = cve_commit.split(',')
                commit_variants = commit_variants.split(', ')
                year = cve.split('-')[1]
                for source_mul_list in os.listdir(source_dir):
                    cve_list_path = os.path.join(source_dir, source_mul_list)
                    if not os.path.isdir(cve_list_path):
                        continue
                    cve_list_path = os.path.join(cve_list_path, year)
                    if not os.path.exists(cve_list_path):
                        continue
                    if cve not in os.listdir(cve_list_path):
                        continue
                    owner_path = os.path.join(cve_list_path, cve)
                    if not os.path.exists(owner_path):
                        continue
                    executor.submit(find_and_process_files, cve, commit, commit_variants, year, owner_path, target_dir)

      
# END: 提取列表
def process_no_slice_data(source_dir,target_dir,copy_path,num_workers=40):
    # 1.copy并尝试预处理数据
    copy_variant_files(source_dir, target_dir,copy_path,num_workers=num_workers)
    # 2.未切片数据打标签

if __name__ == '__main__':
    # 1. 提取CVE和commit列表
    # dataset_path = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20231128/slice20231128/'
    # copy_path = '/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/slice20231128.csv'

    # result = extract_cve_commit_list(dataset_path)
    # save_result_as_txt(result, copy_path)

    # 2. copy 数据
    source_dir = config.Synthetic_path
    # target_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice1215_patch/"
    # copy_path = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/slice1215.csv"

    # target_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20231128_patch/"
    # copy_path = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/slice20231128.csv"

    target_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20240119_patch/"
    copy_path = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/slice20240119.csv"

    process_no_slice_data(source_dir,target_dir,copy_path,num_workers=40)

    
