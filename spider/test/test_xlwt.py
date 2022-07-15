# -*- coding:utf-8 -*-
# @Time : 2022-5-17 1:51
# @Author : zsjng
# @File : test_xlwt.py
# @Software : PyCharm

import xlwt

workbook = xlwt.Workbook(encoding='utf-8')
# 创建workbook对象
worksheet = workbook.add_sheet('sheet1')
# worksheet.write(0,0,'hello')
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f'{i} * {j} = {i * j}', end='\t')
        worksheet.write(i - 1, j - 1, f'{i} * {j} = {i * j}')
    print('\n')
workbook.save('student.xls')
