

## Step3 get corpus
python Step3_get_corpus_many.py [task_name] [选项数字]
- [task_name ] = { full,patch,slice }
- [选项数字] = { 1,2,3,4,5,6,7}
    - 1 去除CVE中重复的内容
    - 2 每一个CVE得到标签为1的数据
    - 3 将标签为1的数据放入文件夹
    - 4 将标签为0的数据放入文件夹 
    - 5 将标签为0的数据放入文件夹
    - 6 删除cve_corpus
    - 7 删除iter_self

## Step4 shuffle data
python Step4_shuffle_data.py [task_name]
- [task_name ] = { full,patch,slice }
## Step6 

python Step5_train_siamese_network.py [task_name]
- [task_name ] = { full,patch,slice }

python Step5_train_siamese_network.py full  >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/train_full_1.log 2>&1
python Step5_train_siamese_network.py patch  >> /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/train_patch_1.log 2>&1

# About relate code 
1. Redebug 
>see the files in the folder /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/ReDeBug

2. SyseVR
>see the files in the url `https://github.com/SeSpProjec/SyseVR-related`,this is a torch verision of SyseVR.