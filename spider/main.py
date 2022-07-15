import requests
from lxml import etree
import re
import csv

from pandas.tseries.frequencies import key


def get_page():
    # 数据的多页爬取，经过观察，所有页面地址中，有一个唯一的参数page_index发生改变
    # 通过对参数page_index的for循环，遍历每一页的页面，实现多页爬取
    for page in range(1, 101):
        url = 'http://search.dangdang.com/?key=python&act=input&page_index=1' + str(
            page + 1) + '#J_tab'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        response = requests.get(url=url, headers=headers)
        parse_page(response)
        # 可以在操作页面实时观察爬取的进度
        print('page %s' % page)


def parse_page(response):
    # 通过etree将图书的七项信息封装为一条数据，保存到data列表当中
    tree = etree.HTML(response.text)
    li_list = tree.xpath('//ul[@class="bigimg"]/li')
    for li in li_list:
        data = []
        try:
            # 通过xpath的方法对所需要的信息进行解析
            # 1、获取书的标题,并添加到列表中
            title = li.xpath('./a/@title')[0].strip()
            data.append(title)
            # 2、获取价格,并添加到列表中
            price = li.xpath('./p[@class="price"]/span[1]/text()')[0]
            data.append(price)
            # 3、获取作者,并添加到列表中
            author = ''.join(li.xpath('./p[@class="search_book_author"]/span[1]//text()')).strip()
            data.append(author)
            # 4、获取出版社
            publis = ''.join(li.xpath('./p[@class="search_book_author"]/span[3]//text()')).strip()
            data.append(publis)
            # 5、获取出版时间,并添加到列表中
            time = li.xpath('./p[@class="search_book_author"]/span[2]/text()')[0]
            pub_time = re.sub('/', '', time).strip()
            data.append(pub_time)
            # 6、获取商品链接,并添加到列表中
            commodity_url = li.xpath('./p[@class="name"]/a/@href')[0]
            data.append(commodity_url)
            # 7、获取评论数量，并添加到列表中
            comment = li.xpath('./p[@class="search_star_line"]/a/text()')[0].strip()
            data.append(comment)
        except:
            pass
