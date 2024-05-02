## Step3 get corpus
python Step3_get_corpus_many.py [task_name] [option_number]
- [task_name ] = { full,patch,slice }
- [option_number] = { 1,2,3,4,5,6,7}
    - 1 Remove duplicate content from CVE
    - 2 Get data with label 1 for each CVE
    - 3 Put data with label 1 into a folder
    - 4 Put data with label 0 into a folder 
    - 5 Put data with label 0 into a folder
    - 6 Delete cve_corpus
    - 7 Delete iter_self

## Step4 shuffle data
python Step4_shuffle_data.py [task_name]
- [task_name ] = { full,patch,slice }
## Step6 train siamese network

python Step6_train_siamese_network.py [task_name]
- [task_name ] = { full,patch,slice }

nohup python Step6_train_siamese_network.py full  > /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/train_full_1.log 2>&1 &
nohup python Step6_train_siamese_network.py patch  > /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/train_patch_1.log 2>&1 &
nohup python Step6_train_siamese_network.py slice  > /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/train_slice_1.log 2>&1 &

# About related Code
## RQ3 Ablation Evaluation



## RQ4 Comparative Effectiveness

1. Redebug 
>see the files in the folder /home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/ReDeBug

2. SyseVR
>see the files in the url [SyseVR_re](https://github.com/SeSpProjec/SyseVR-related),this is a torch verision of SyseVR.