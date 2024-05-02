import os
import shutil
import random

def copy_random_variant(source_base_dir, target_base_dir, filename="modify_slice_norm.txt"):
    """遍历源目录，对每个commit随机选择一个变种目录，复制指定文件到目标目录。"""
    for cve_dir in os.listdir(source_base_dir):  # 遍历每个CVE目录
        cve_path = os.path.join(source_base_dir, cve_dir)
        if not os.path.isdir(cve_path):
            continue
        
        for commit_dir in os.listdir(cve_path):  # 遍历每个commit目录
            commit_path = os.path.join(cve_path, commit_dir)
            if not os.path.isdir(commit_path):
                continue
            
            variant_dirs = [d for d in os.listdir(commit_path) if os.path.isdir(os.path.join(commit_path, d))]
            if not variant_dirs:
                continue
            
            # 随机选择一个变种目录
            selected_variant = random.choice(variant_dirs)
            variant_path = os.path.join(commit_path, selected_variant)
            file_path = os.path.join(variant_path, filename)
            
            if os.path.exists(file_path):
                # 设置目标文件路径
                relative_path = os.path.relpath(variant_path, source_base_dir)
                target_path = os.path.join(target_base_dir, relative_path, filename)
                
                # 确保目标目录存在
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # 复制文件
                shutil.copy(file_path, target_path)
                print(f"Copied {file_path} to {target_path}")

def main():
    # source_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20231128/slice20231128"
    # target_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20231128_slicing"

    # source_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20240119/"
    # target_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice20240119_slicing/"

    source_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice1215/slice/"
    target_base_dir = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice1215_slicing/"
    copy_random_variant(source_base_dir, target_base_dir)

if __name__ == "__main__":
    main()
