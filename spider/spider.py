# -*- coding:utf-8 -*-
# @Time : 2022-5-16 15:29
# @Author : zsjng
# @File : spider.py
# @Software : PyCharm
import re
import urllib.request, urllib.error
from bs4 import BeautifulSoup
import xlwt

"""爬取网页,逐一解析,保存数据"""


def main():
    baseurl = 'https://movie.douban.com/top250?start='
    datalist = get_data(baseurl)
    savepath = '.\\豆瓣电影Top250.xls'
    save_data(datalist, savepath)

    # ask_url(baseurl)


# <a href="https://movie.douban.com/subject/1292052/">
find_link = re.compile(r'<a href="(.*?)"')
find_img = re.compile(r'<img.*src="(.*?)"', re.S)
find_title = re.compile(r'<span class="title">(.*)</span>')
# <span class="rating_num" property="v:average">
find_rate = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
find_judge = re.compile(r'<span>(\d*)人评价</span>')
find_inq = re.compile(r'<span class="inq">(.*)</span>')
find_bd = re.compile(r'<p class="">(.*?)</p>', re.S)


# 影片链接的规则
# 创建正则表达式对象

def get_data(baseurl):
    datalist = []
    for i in range(0, 10):  # 调用获取页面信息的函数10次,每次25条
        url = baseurl + str(i * 25)
        html = ask_url(url)
        # 保存获取到的网页源码

        soup = BeautifulSoup(html, 'html.parser')
        for item in soup.find_all('div', class_='item'):
            data = []
            item = str(item)
            link = re.findall(find_link, item)[0]
            data.append(link)

            img = re.findall(find_img, item)[0]
            data.append(img)

            title = re.findall(find_title, item)
            if len(title) == 2:
                c_title = title[0]  # 中文名
                data.append(c_title)
                o_title = title[1].replace('/', '')
                data.append(o_title)  # 外文名
            else:
                data.append(title[0])
                data.append(' ')  # 没有英文名就留空

            rate = re.findall(find_rate, item)[0]
            data.append(rate)

            judge_num = re.findall(find_judge, item)[0]
            data.append(judge_num)

            inq = re.findall(find_inq, item)
            if len(inq) != 0:
                inq = inq[0].replace('。', '')
                data.append(inq)
            else:
                data.append(' ')

            bd = re.findall(find_bd, item)[0]
            bd = re.sub('<br(\s+)?/>(\s+)?', ' ', bd)
            bd = re.sub('/', ' ', bd)
            data.append(bd.strip())

            datalist.append(data)
            # 去掉前后空格

        # print(datalist)

    return datalist


def ask_url(url):
    head = {  # 模拟浏览器头部信息,向豆瓣服务器发送消息
        'User-Agent': 'Mozilla / 5.0(Windows NT 10.0;Win64;x64) AppleWebKit / 537.36(KHTML, likeGecko) Chrome / 101.0.4951.64Safari / 537.36Edg / 101.0.1210.47'
    }
    # 用户代理,表示告诉豆瓣服务器,我们是什么类型的机器和浏览器(本质上 是告诉浏览器 ,我们可以接受什么水平的文件内容)

    request = urllib.request.Request(url, headers=head)
    html = ''
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode('utf-8')
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, 'code'):
            # 是否包含这个属性 has attribute
            print(e.code)
        if hasattr(e, 'reason'):
            print(e.reason)

    return html


def save_data(datalist, savepath):
    print('save...')
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # 创建workbook对象
    sheet = book.add_sheet('豆瓣电影', cell_overwrite_ok=True)
    col = ('电影详情链接', '图片链接', '影片中文名', '影片外文名', '评分', '评价数', '概况', '相关信息')
    for i in range(0, 8):
        sheet.write(0, i, col[i])
    # 写入第一行
    for i in range(0, 250):
        print(f'第{i+1}条')
        data = datalist[i]
        for j in range(0, 8):
            sheet.write(i + 1, j, data[j])
    book.save(savepath)
    print('爬取完毕')


if __name__ == '__main__':
    main()
