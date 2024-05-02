
# -*- coding: utf-8 -*-

import os
import re
from multiprocessing import Pool

from numpy import size

def filter_line_numbers(text, special_marker='???'):# 这个函数的功能是为了过滤配对数据中的行号
    import re
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)

    # 使用正则表达式进行分词
    parts = TOKENIZER_RE.findall(text)
    filtered_parts = []

    for part in parts:
        # 检查是否包含特殊标记
        if special_marker in part:
            filtered_parts.append(part)
        else:
            # 如果部分不是仅由数字组成，则保留
            if not re.fullmatch(r'\d+', part):
                filtered_parts.append(part)

    # 重新组合处理过的部分
    filtered_text = ' '.join(filtered_parts)

    return filtered_text

def rm_line_numebr(data): # 这个函数的功能是为了去除数据中的行号
    # 使用正则表达式匹配每一行末尾的数字并移除
    # 表达式解释：\d+ 匹配一个或多个数字，$ 表示行末
    # re.MULTILINE 让 ^ 和 $ 匹配每一行的开头和结尾
    # 使用正则表达式匹配每一行末尾的数字并移除
    import re
    cleaned_lines = []
    for line in data.splitlines():
        cleaned_line = re.sub(r'\d+$', '', line)
        cleaned_lines.append(cleaned_line)
    # 返回处理后的数据，每行之间用空格连接
    return ' '.join(cleaned_lines)


def remove_cve_in_directory(directory_info):
    """处理单个目录中的文件删除任务"""
    data_path, files = directory_info
    for file in files:
        if re.match(r'cve_corpus.txt', file):
            os.remove(os.path.join(data_path, file))
            print("Deleted " + data_path + "/" +file)

def rm_cve_corpus(data_path): # 这个函数的功能是为了删除指定路径下的 cve_corpus.txt 文件
    """使用多进程删除指定路径下的 cve_corpus.txt 文件"""
    with Pool(processes=20) as pool:  # 使用4个进程，可以根据需要调整
        all_files = []
        for root, dirs, files in os.walk(data_path):
            all_files.append((root, files))
        pool.map(remove_cve_in_directory, all_files)  # map函数分配任务到进程池中的进

def remove_itself_in_directory(directory_info):
    """处理单个目录中的文件删除任务"""
    data_path, files = directory_info
    for file in files:
        if re.match(r'iter_self.txt', file):
            os.remove(os.path.join(data_path, file))
            print("Deleted " + data_path + "/" +file)

def rm_itself_file(data_path): # 这个函数的功能是为了删除指定路径下的自身文件
    import re
    import os

    with Pool(processes=20) as pool:  # 使用4个进程，可以根据需要调整
        all_files = []
        for root, dirs, files in os.walk(data_path):
            all_files.append((root, files))
        pool.map(remove_itself_in_directory, all_files)  # map函数分配任务到进程池中的进

def comments_Eliminate(patch_file_content):#删除注释并保存文件
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

def filter_lines(lines, keywords):
    """
    过滤掉包含任何指定关键字的行。
    
    :param lines: 包含字符串的列表，每个元素代表一行文本。
    :param keywords: 关键字列表，用于过滤含有这些关键字的行。
    :return: 过滤后的行列表。
    """
    # 使用列表推导式过滤行
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in keywords)]
    print
    return filtered_lines

# 规定补丁的行数补丁大于100行的补丁，大于100行的补丁会被跳过
def remove_patch(patch_content):
    keywords = ["..", "--","@@","++","---","+++"]
    filtered_lines = filter_lines(patch_content, keywords)
    # 删除注释
    filtered_lines = comments_Eliminate(filtered_lines)
    return filtered_lines

def filter_patch(patch_content):

    if len(patch_content) >= 400:
        return True
    return False

