# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
from os import path
from multiprocessing import Process
# import  configuration as config
# from helpers import helper_zz
def comments_Eliminate(patch_file_content, file2_path):#删除注释并保存文件
    """
    删除代码中的注释，并返回去除注释后的代码行列表。
    
    :param patch_file_content: 包含代码的字符串列表，每个元素是一行代码。
    :return: 删除注释后的代码行列表。
    """
    result = []  # 结果列表，用来存储删除注释后的代码
    flag = 0  # 标记位，标记是否进入多行注释/* */
    for code in patch_file_content:
        clean_line = ''  # 用来构建当前处理行的未注释内容
        i = 0  # 字符索引
        while i < len(code):
            # 单行注释//
            if flag == 0 and i + 1 < len(code) and code[i] == '/' and code[i + 1] == '/':
                break  # 跳出循环，忽略此行剩余部分
            # 多行注释开始/* .....  */
            elif flag == 0 and i + 1 < len(code) and code[i] == '/' and code[i + 1] == '*':
                flag = 1
                i += 1  # 跳过'*'
            # 多行注释结束*/
            elif flag == 1 and i + 1 < len(code) and code[i] == '*' and code[i + 1] == '/':
                flag = 0
                i += 1  # 跳过'/'
            # 如果在多行注释中，跳过当前字符
            elif flag == 1:
                pass
            # 如果不在注释中，添加字符到clean_line
            elif flag == 0:
                clean_line += code[i]
            i += 1
        # 如果clean_line不为空，添加到结果列表
        if clean_line and not flag:
            result.append(clean_line)
    return result