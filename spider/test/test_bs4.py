# -*- coding:utf-8 -*-
# @Time : 2022-5-16 15:49
# @Author : zsjng
# @File : test_bs4.py
# @Software : PyCharm

"""
bs4 将复杂的HTML文档转换成树形结构,所有的对象可以归纳为4种:
- Tag #标签及其内容
- NavigableString #标签里的内容 字符串
- BeautifulSoup #整个文档 自身
- Comment #注释 是一个特殊的NavigableString,输出的内容不包括注释符号
"""
from bs4 import BeautifulSoup

file = open('../baidu.html', 'rb')
html = file.read()
bs = BeautifulSoup(html, 'html.parser')
# print(bs.title.string,type(bs.title.string))
print(bs.a,type(bs.a))
print(bs.a.string,type(bs.a.string))