# -*- coding:utf-8 -*-
# @Time : 2022-5-17 2:16
# @Author : zsjng
# @File : test_sqlite.py
# @Software : PyCharm

import sqlite3

""" 链接数据库"""
# conn = sqlite3.connect('test.db')
#
# print('成功打开数据库')
#
# c = conn.cursor()
# # 获取游标
# sql = '''
#     create table company
#         (id int primary key not null,
#         name text not null,
#         age int not null,
#         address char(50),
#         salary real);
#
#
#
# '''
#
# c.execute(sql)
# conn.commit()
# # 提交操作
# conn.close()
# # 关闭数据库
# print('成功建表')

# 3.插入数据
# conn = sqlite3.connect('test.db')
#
# print('成功打开数据库')
#
# c = conn.cursor()
# # 获取游标
# sql1 = '''
#     insert into company (id,name,age,address,salary)
#     values (1,'张三',32,'成都',8000)
#
#
#
# '''
# sql2 = '''
#     insert into company (id,name,age,address,salary)
#     values (2,'李四',30,'成都',15000)
#
#
#
# '''
# c.execute(sql1)
# c.execute(sql2)
# conn.commit()
# # 提交操作
# conn.close()
# # 关闭数据库
# print('成功建表')
#

# 4.查询数据库

conn = sqlite3.connect('test.db')

print('成功打开数据库')
c = conn.cursor()

sql = 'select id,name,address,salary from company'

cursor = c.execute(sql)

for row in cursor:
    print('id=', row[0])
    print('name=', row[1])
    print('address=', row[2])
    print('salary=', row[3], '\n')
# 获取游标
conn.commit()
# 提交操作
conn.close()
# 关闭数据库
print('成功建表')
