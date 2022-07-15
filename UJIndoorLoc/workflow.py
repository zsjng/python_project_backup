import os
import sys
import time

# 每次运行前删除上次运行后的 log.txt 和 tmp.txt 文件
lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # 在这里面写你要的参数 这里写的是 从128-321，每隔32取一次值的列表
exp_ids = [20220509, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10折交叉验证
c = 1
sum = 0
for lr in lrs:  # 从这里依次读取这个列表里的值 为 hd
    total = int(len(exp_ids)*len(lrs))
    for exp_id in exp_ids:
        print('\nProgress {}/{}'.format(c, total))
        start = time.time()
        os.system('python run-E2E.py --seed {} --epochs 50 --lr {} >tmp.txt'.format(exp_id, lr))  # 在这个命令里以这两个参数运行程序
        with open("tmp.txt", "r") as file:
            lines = file.readlines()
            with open("log.txt", "a") as f:
                f.write('seed:{}  learning_rate:{}\n'.format(exp_id, lr))
                f.write(lines[-1])  # 将最后一行打印结果写入
                f.write('\n')
        end = time.time()
        times = round((end-start)/60) if (end-start) > 60 else (end-start)/60
        print('It takes {:.1f} mins'.format(times))
        sum += times
        if c != len(exp_ids):
            ava = sum/c
            rest = ava*(total-c)
            print('Estimated remaining time: {:.1f} mins.'.format(rest))  # It may be inaccurate bacause of database discrepancies.
        c += 1
sys.exit()
