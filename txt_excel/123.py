# -*- coding:utf-8 -*-
# @Time : 2022-5-18 14:17
# @Author : zsjng
# @File : 123.py
# @Software : PyCharm
import xlwt
test=[]
result = []
test_new=[]
with open('right.txt','r',encoding='utf-8') as f:
    for line in f.readlines():  # readlines以列表输出文件内容
        line = line.replace("\n", "").replace("\n", "")  # 改变元素，去掉，和换行符\n,tab键则把逗号换成"/t",空格换成" "
        result.append(line)
for i in result:
    # print(i)
    test.append(i.split(','))
# print(test)
for i in test:
    i_new = []
    for j in i:
        j = j.replace('SHAP values for ll: mean-[','')
        j = j.replace('positive mean [', '')
        j = j.replace('number positives [', '')
        j = j.replace('number negatives [', '')
        j = j.replace('negative mean [', '')
        j = j.replace('SHAP values for lh: mean-[', '')
        j = j.replace('SHAP values for hl: mean-[', '')
        j = j.replace('SHAP values for hh: mean-[', '')
        j = j.replace(']', '')

        i_new.append(j)
    test_new.append(i_new)
print(test_new[0])
def save_data(datalist, savepath):
    print('save...')
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # 创建workbook对象
    sheet = book.add_sheet('file', cell_overwrite_ok=True)
    col = ('file_name', 'SHAP values for ll', 'positive mean', 'number positives', 'negative mean', 'number negatives', 'file_name', 'SHAP values for lh', 'positive mean', 'number positives', 'negative mean', 'number negatives', 'file_name', 'SHAP values for hl', 'positive mean', 'number positives', 'negative mean', 'number negatives', 'file_name', 'SHAP values for hh', 'positive mean', 'number positives', 'negative mean', 'number negatives')
    for i in range(0, 24):
        sheet.write(0, i, col[i])
    # 写入第一行
    for i in range(0, 1999,2):
        print(f'第{i+1}条')
        data = datalist[i]
        for j in range(0, 24):
            sheet.write(i + 1, j, data[j])
    book.save(savepath)

save_data(test_new,'./test.xls')