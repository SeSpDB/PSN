# -*- coding: utf-8 -*-
#!/usr/bin/env python

# 
import os
from ssl import VERIFY_X509_TRUSTED_FIRST
def extract_version_info(path,key_word):
    # 将路径分割成列表
    parts = path.split('/')
    
    # 找到 'equal_func_patch' 的索引
    try:
        idx = parts.index(key_word)
    except ValueError:
        # 如果路径中不存在 'equal_func_patch'
        return False

    # 检查索引是否足够向上回溯两层
    if idx < 2:
        return False # 找到了key words

    # 提取版本信息
    version_info = parts[idx-2]
    return True

def find_variants(root_dir, variant_name):
    """
    在 root_dir 及其所有子目录中寻找名为 variant_name 的文件夹。
    返回包含这些文件夹的完整路径列表。
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if variant_name in dirnames:
            paths.append(os.path.join(dirpath, variant_name))
    if len(paths) == 0:
        return False
    return paths