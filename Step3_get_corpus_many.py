# coding=utf-8


import string
import os
import itertools
import random
import sys


# from networkx import group_in_degree_centrality
from data_process import help
import threading
from data_process import mul_thread
import os
from concurrent.futures import ThreadPoolExecutor

def run_thread(func, *args):
    thread = threading.Thread(target=func, args=args)
    thread.start()
    return thread

def getCVEcorpus(CVE_dirs):
    
    for every_CVE in CVE_dirs:
        print("Processing ",every_CVE)
        CVE_path = os.path.join(SOURCE_CVE_PATH, every_CVE)
        # 在每个CVE下面创建切片
        CVE_slice_path = os.path.join(CVE_path, 'cve_corpus.txt')
        if os.path.exists(CVE_slice_path):
            os.remove(CVE_slice_path)
            os.mknod(CVE_slice_path)
        FUN_dirs = os.listdir(CVE_path)

        CVE_corpus_list = []
        #  every_FUN是CVE的每一个commit
        for every_FUN in FUN_dirs: # commit 
            if '.txt' in every_FUN:
                continue
            FUN_path = os.path.join(CVE_path, every_FUN)
            patch_and_fun_path = os.listdir(FUN_path)
            #  patch_and_fun_path  为commit下面自定义文件夹的集合  （如：lable）的列表     example    ['lable','NVD']
            for every_patch_and_fun_path in patch_and_fun_path: # VAR
                # if "LABEL" not in every_patch_and_fun_path:
                #     continue
                if ".patch" in every_patch_and_fun_path: # 处理patch文件
                    every_patch_and_fun_path_list = [every_patch_and_fun_path]
                    
                    file_path = os.path.join(FUN_path, every_patch_and_fun_path)  # 只使用这个进行打标签
                    f = open(file_path, 'r')
                    temp_contents = f.readlines()
                    f.close()
                    line_str = ''
                    if help.filter_patch(temp_contents):
                        continue
                    temp_contents = help.remove_patch(temp_contents)

                    line_str = help.rm_line_numebr("".join(temp_contents))
                    line_str += '\n'
                    CVE_corpus_list.append(line_str)

                else:                               # 处理切片后的文件
                    every_patch_and_fun_path = os.path.join(FUN_path, every_patch_and_fun_path)
                    every_patch_and_fun_path_list = os.listdir(every_patch_and_fun_path)
                    for every_final_file in every_patch_and_fun_path_list:
                        if "modify_slice_norm.txt" !=  every_final_file: # 只对这个执行，其他的不执行
                            continue 
                        file_path = os.path.join(every_patch_and_fun_path, every_final_file)  # 只使用这个进行打标签
                        f = open(file_path, 'r')
                        temp_contents = f.readlines()
                        f.close()
                        line_str = ''
                        if help.filter_patch(temp_contents):
                            continue
                        temp_contents = help.remove_patch(temp_contents)
                        line_str = help.rm_line_numebr("".join(temp_contents))
                        line_str += '\n'
                        CVE_corpus_list.append(line_str)
        CVE_corpus_dict = {}
        x = 0
        for every in CVE_corpus_list:
            if every not in CVE_corpus_dict.values():
                CVE_corpus_dict[x] = every
                x = x + 1
        CVE_corpus_list1 = []
        for every in CVE_corpus_dict:
            CVE_corpus_list1.append(CVE_corpus_dict[every])
        f = open(CVE_slice_path, 'w')
        for every_patch in CVE_corpus_list1:
            f.write(every_patch)
        f.close() # 生成了CVE的语料库

def get_cve_corpus(CVE_PATH, max_workers=20):
    CVE_dirs = [os.path.join(CVE_PATH, d) for d in os.listdir(CVE_PATH) if os.path.isdir(os.path.join(CVE_PATH, d))]
    # 根据max_workers来分批处理
    batch_size = len(CVE_dirs) // max_workers + (len(CVE_dirs) % max_workers > 0)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 切分CVE_dirs成多个批次
        for i in range(0, len(CVE_dirs), batch_size):
            cve_batch = CVE_dirs[i:i + batch_size]
            executor.submit(getCVEcorpus, cve_batch)

def iter_self_fun(iter_self_list, pair_limit):
    iter_list = []
    not_scan_list = []
    
    for every in iter_self_list:
        not_scan_list.append(every)
        # 创建过滤列表，仅包含还未扫描过的元素
        available_pairs = [x for x in iter_self_list if x not in not_scan_list]
        
        # 确保抽样数量不会超过列表长度
        sample_size = min(pair_limit, len(available_pairs))
        if sample_size > 0:  # 确保有可用的配对项
            random_pairs = random.sample(available_pairs, sample_size)
            for every_other in random_pairs:
                temp_str = every + " ??? " +every_other+ " ??? 1\n'"
                iter_list.append(temp_str)
        else:
            continue  # 如果没有可用的配对项，继续下一个元素

    return iter_list

def iter_self(CVE_PATH,pair_limit=20):
    iter_self_list = []
    CVE_dirs = os.listdir(CVE_PATH)
    for every_CVE in CVE_dirs:
        CVE_path = os.path.join(CVE_PATH, every_CVE)
        CVE_slice_path = os.path.join(CVE_path, 'cve_corpus.txt')
        iter_self_path = os.path.join(CVE_path, 'iter_self.txt')
        f = open(CVE_slice_path, 'r')
        CVE_corpus_list = f.readlines()
        f.close()
        CVE_corpus_list1 = []
        for every in CVE_corpus_list:
            every = every.strip()
            CVE_corpus_list1.append(every)
        iter_self_list = iter_self_fun(CVE_corpus_list1,pair_limit)
        f = open(iter_self_path, 'w')
        for everyline in iter_self_list:
            f.write(everyline)
        f.close()
        print(every_CVE,' okok')

def process_itself_cve(cve_path, pair_limit):
    CVE_slice_path = os.path.join(cve_path, 'cve_corpus.txt')
    iter_self_path = os.path.join(cve_path, 'iter_self.txt')
    
    with open(CVE_slice_path, 'r') as f:
        CVE_corpus_list = f.readlines()
    
    CVE_corpus_list1 = [line.strip() for line in CVE_corpus_list]
    iter_self_list = iter_self_fun(CVE_corpus_list1, pair_limit)
    
    with open(iter_self_path, 'w') as f:
        for everyline in iter_self_list:
            f.write(everyline)
    
    print(os.path.basename(cve_path)+ "okok")

def iter_self_parallel(CVE_PATH, pair_limit=10, max_workers=10):
    CVE_dirs = [os.path.join(CVE_PATH, d) for d in os.listdir(CVE_PATH) if os.path.isdir(os.path.join(CVE_PATH, d))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cve_dir in CVE_dirs:
            executor.submit(process_itself_cve, cve_dir, pair_limit)

def get_1_corpus(CVE_PATH, output_1_path,save_path_list):
    global txt_lable_1_i
    CVE_dirs = os.listdir(CVE_PATH)
    corpus_path_1 = os.path.join(output_1_path, save_path_list +'_train_input_lable1_{}.txt'.format(txt_lable_1_i))
    if not os.path.exists(corpus_path_1):
        f = open(corpus_path_1, 'w')
        f.close()
    for every_CVE in CVE_dirs:
        if '.txt' in every_CVE:
            continue
        CVE_path = os.path.join(CVE_PATH, every_CVE)
        iter_self_path = os.path.join(CVE_path, 'iter_self.txt')
        f = open(iter_self_path, 'r')
        temp_lines = f.readlines()
        f.close()
        size = os.path.getsize(corpus_path_1)
        if size != 0:
            if size > 1024 * 1024 * 100:
                txt_lable_1_i += 1
                corpus_path_1 = os.path.join(output_1_path, save_path_list +'_train_input_lable1_{}.txt'.format(txt_lable_1_i))
                if not os.path.exists(corpus_path_1):
                    f = open(corpus_path_1, 'w')
                    f.close()
        f = open(corpus_path_1, 'a')
        for every in temp_lines:
            f.write(every)
        f.close()
        print("lable 1 : This CVE done!_",every_CVE)


def get_0_corpus(base_paths, line_value, output_0_path, groups,save_path_list):
    global txt_lable_0_i
    txt_lable_0_i = 0
    corpus_path_0 = os.path.join(output_0_path, save_path_list + "_train_input_lable0_" + str (txt_lable_0_i) + ".txt")
    if not os.path.exists(corpus_path_0):
        with open(corpus_path_0, 'w') as f:
            f.close()

    # 为每个组准备一个完整的CVE列表路径
    cve_paths = []
    for group in groups:
        group_path = os.path.join(base_paths, group)
        if group=="slice1215":
            group_path = os.path.join(group_path,"slice")
        elif group=="slice20231128":
            group_path = os.path.join(group_path,"slice20231128")
        CVE_dirs = [os.path.join(group_path, d) for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        cve_paths.extend(CVE_dirs)

    random.shuffle(cve_paths)

    for i in range(len(cve_paths)):
        cve1_path = cve_paths[i]
        cve_corpus_path1 = os.path.join(cve1_path, 'cve_corpus.txt')
        if not os.path.isfile(cve_corpus_path1):
            continue
        with open(cve_corpus_path1, 'r') as f:
            list1 = [line.strip() for line in f.readlines()]

        for j in range(i + 1, len(cve_paths)):
            cve2_path = cve_paths[j]
            cve_corpus_path2 = os.path.join(cve2_path, 'cve_corpus.txt')
            if not os.path.isfile(cve_corpus_path2):
                continue
            with open(cve_corpus_path2, 'r') as f:
                list2 = [line.strip() for line in f.readlines()]


            cve1 = cve1_path.split('/')[-1]
            cve2 = cve2_path.split('/')[-1]
            num = min(line_value, len(list2))
            random.shuffle(list1)
            random.shuffle(list2)
            match_count = 0
            for x in itertools.product(list1[:num], list2[:num]):
                if match_count >= line_value:
                    break
                str_temp = cve1 + " " +cve2 + x[0] + "???" + x[1] + " ??? 0\n" # 加入cve是为了判定是哪几个cve没有进行正则化
                size = os.path.getsize(corpus_path_0)
                if size > 1024 * 1024 * 100:  # 当文件大于100MB时，创建新文件
                    txt_lable_0_i += 1
                    corpus_path_0 = os.path.join(output_0_path, save_path_list + "_train_input_lable0_" + str(txt_lable_0_i) +".txt")
                    if not os.path.exists(corpus_path_0):
                        with open(corpus_path_0, 'w') as f:
                            f.close()
                with open(corpus_path_0, 'a') as f:
                    f.write(str_temp)
                match_count += 1

            print("lable 0 : done!" + os.path.basename(cve2_path))


def add_corpus():
    corpus1_path = r'E:\my_LSTM\deep-siamese-text-similarity-master\train_snli\testdata\corpus_1.txt'
    corpus0_path = r'E:\my_LSTM\deep-siamese-text-similarity-master\train_snli\testdata\corpus_0.txt'
    final_path = r'E:\my_LSTM\deep-siamese-text-similarity-master\train_snli\testdata\slice_corpus.txt'
    f = open(corpus1_path, 'r')
    temp_list = f.readlines()
    f.close()
    f = open(final_path, 'a')
    for every in temp_list:
        f.write(every)
    f.close()
    print('done!')
    f = open(corpus0_path, 'r')
    temp_list = f.readlines()
    f.close()
    f = open(final_path, 'a')
    for every in temp_list:
        f.write(every)
    f.close()
    print('done!')


if __name__ == "__main__":

    # 以下每一个函数取消注释各自运行一遍
    #   CVE_PATH存放切片路径
    # pair_1_counts = 5
    # pair_0_counts = 2
    # SAVE_FLAG = 'slice1215'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG + "/slice/"
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 5
    # pair_0_counts = 2
    # SAVE_FLAG = 'slice1215_slicing'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 10
    # pair_0_counts = 5
    # SAVE_FLAG = 'slice1215_patch'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"


    # pair_1_counts = 50
    # pair_0_counts = 50
    # SAVE_FLAG = 'slice20231128'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG + "/slice20231128/"
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 50
    # pair_0_counts = 50
    # SAVE_FLAG = 'slice20231128_slicing'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG 
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 80
    # pair_0_counts = 80
    # SAVE_FLAG = 'slice20231128_patch'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG 
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 50
    # pair_0_counts = 50
    # SAVE_FLAG = 'slice20240119'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG 
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 50
    # pair_0_counts = 50
    # SAVE_FLAG = 'slice20240119_slicing'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG 
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"

    # pair_1_counts = 70
    # pair_0_counts = 70
    # SAVE_FLAG = 'slice20240119_patch'
    # CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + SAVE_FLAG 
    # SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + SAVE_FLAG + "/"


    # 考虑到可能会出现负样本过少的问题，所以需要将所有类似的都合并起来，然后进行训练
    pair_1_counts = 30
    pair_0_counts = 40
    # task = "patch" # full patch slice
    task = sys.argv[1]

    if task == "full":
        group = ['slice1215', 'slice20231128', 'slice20240119']
    elif task == "patch":
        group = ['slice1215_patch', 'slice20231128_patch', 'slice20240119_patch']
    else:   # silice
        group = ['slice1215_slicing', 'slice20231128_slicing', 'slice20240119_slicing']
    
    for save_path_list in group:
        SOURCE_CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/' + save_path_list
        SAVE_PAIR_PATH = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/" + task + "/" # full patch  silice 存储在大类文件夹中

        if os.path.exists(SAVE_PAIR_PATH)!=True:
            os.makedirs(SAVE_PAIR_PATH)

        global match_limit # 控制cve与其他cve负样本匹配的次数
        global txt_lable_1_i
        global txt_lable_0_i
        match_limit = pair_0_counts # 只运行每一个commit与其他commit 匹配1次
        txt_lable_1_i = 0
        txt_lable_0_i = 0
        output_1_path = SAVE_PAIR_PATH
        output_0_path = SAVE_PAIR_PATH
        if save_path_list=="slice1215":
            SOURCE_CVE_PATH = os.path.join(SOURCE_CVE_PATH,"slice")
        elif save_path_list=="slice20231128":
            SOURCE_CVE_PATH = os.path.join(SOURCE_CVE_PATH,"slice20231128")

        func_select = sys.argv[2] # 选项 
        if func_select == '1': # 去除CVE中重复的内容
            get_cve_corpus(SOURCE_CVE_PATH)
            # tasks = [
            #     {'function': getCVEcorpus, 'args': (CVE_PATH,)} # ()中的参数是函数的参数
            # ]
            # mul_thread.multi_thread_scheduler(tasks)

        elif func_select == '2': #  每一个CVE得到标签为1的数据
            # iter_self_parallel(CVE_PATH, pair_limit=pair_1_counts, max_workers=10)
            iter_self(SOURCE_CVE_PATH,pair_limit = pair_1_counts)

        elif func_select == '3': # 将标签为1的数据放入文件夹
            get_1_corpus(SOURCE_CVE_PATH, output_1_path,save_path_list)

        elif func_select == '4':    # 将标签为0的数据放入文件夹 
            SOURCE_CVE_PATH = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/'
            get_0_corpus(SOURCE_CVE_PATH, match_limit, output_0_path,group,save_path_list) # group 是希望在所有的CVE中进行负样本的配对
            break

        elif func_select == '5':   # 将标签为0的数据放入文件夹
            add_corpus()

        elif func_select == '6':   # 删除cve_corpus
            help.rm_cve_corpus(SOURCE_CVE_PATH)

        elif func_select == '7':   # 删除iter_self
            help.rm_itself_file(SOURCE_CVE_PATH)

        else:
            print('error!')

    print('done!')




