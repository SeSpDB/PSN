#! /usr/bin/env python
#coding=utf-8
import threading

def thread_function(name, delay):
    import time
    print("Thread "+name+ ": starting")
    time.sleep(delay)
    print("Thread "+name+ ": finishing after " + delay +": seconds")

def multi_thread_scheduler(tasks):
    threads = []
    for task in tasks:
        # 每个task是一个字典，包含函数和参数
        thread = threading.Thread(target=task['function'], args=task['args'])
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    tasks = [
        {'function': thread_function, 'args': ("Thread-1", 2)},
        {'function': thread_function, 'args': ("Thread-2", 4)}
    ]
    multi_thread_scheduler(tasks)