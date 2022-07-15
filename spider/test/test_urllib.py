# -*- coding:utf-8 -*-
# @Time : 2022-5-16 5:27
# @Author : zsjng
# @File : test_urllib.py
# @Software : PyCharm


import urllib.request, urllib.error

# # 获取一个get请求
# res = urllib.request.urlopen('http://www.baidu.com')
# print(res.read().decode('utf-8'))
# # 对网页源码进行utf-8解码

# post方法测试

import urllib.parse

# data = bytes(urllib.parse.urlencode({'hello':'world'}),encoding='utf-8')
# try:
#     res = urllib.request.urlopen('http://www.baidu.com')
#     print(res.read().decode('utf-8'))
#     print(res.getheader("Server"))
# except urllib.error.URLError as e:
#     print('time out')
# url = 'http://httpbin.org/post'
#
# headers = {
#     'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36 Edg/101.0.1210.47'
# }
# data = bytes(urllib.parse.urlencode({'name': 'Tom'}), encoding='utf-8')
# req = urllib.request.Request(url=url, data=data, headers=headers, method='POST')
# response = urllib.request.urlopen(req)
# print(response.read().decode('utf-8'))

url = 'https://www.douban.com'
headers = {
    'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36 Edg/101.0.1210.47'
}
req = urllib.request.Request(url=url,headers=headers)
response = urllib.request.urlopen(req)
print(response.read().decode('utf-8'))