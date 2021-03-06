# -*- coding:utf-8 -*-
# @Time : 2022-5-18 21:48
# @Author : zsjng
# @File : p.py
# @Software : PyCharm
import statsmodels.api as sm
import pandas as pd
# 提供对许多不同统计模型估计的类和函数，并且可以进行统计测试和统计数据的探索。
# sklearn是基于python语言的机器学习工具包，是目前做机器学习项目当之无愧的第一工具。
# sklearn自带了大量的数据集，可供我们练习各种机器学习算法。 s
# klearn集成了数据预处理、数据特征选择、数据特征降维、分类\回归\聚类模型、模型评估等非常全面算法。

# 导入数据

excelFile = r'C:\Users\chen\PycharmProjects\建模\hs300因子优化1710-2203模型.xlsx'
df = pd.DataFrame(pd.read_excel(excelFile, sheet_name='建模数据'))
# 将数据导入df

# 导出自变量因变量
x = df[[
    'PMI（单位：%）',
    '工业增加值:当月同比（单位：%)',
    '预测平均值:CPI:当月同比',
    'CPI:当月同比',
    '预测平均值:PPI:当月同比',
    'PPI:全部工业品:当月同比',
    'M2:同比',
    '社融同比增速（单位：%）',
    '预测社融同比增速（单位：%）',
    '逆回购利率:7天',
    '中期借贷便利(MLF)变化率',
    'CFETS人民币汇率指数变化率',
    '北向资金流入金额变化率',
    '10年期国债收益率',
    '美元指数（DINIV)',
    '10年期国债利率-1年期国债利率',
    'IF股权风险溢价',
    '美国:国债实际收益率:10年期',
    '美国:CPI:当月同比',
    '美国非农就业环比增速',
    '30大中城市:商品房成交面积',
    '库存:主要钢材品种:合计',
    '波罗的海干散货指数(BDI)',
    '韩国出口：同比增速',
    '市净率倒数（BP）',
    '市盈率导数',
    '市盈率倒数（EP)增速平方',
    '净资产收益率ROE(平均)',
    '总资产净利率ROA',
    '销售毛利率',
    '销售净利率',
    '营业收入(同比增长率)',
    '归属母公司股东的净利润(同比增长率)',
    '经营活动产生的现金流量净额(同比增长率)',
    '净资产收益率(摊薄)(同比增长率)',
    '总市值增速',
    '总市值对数',
    '成交量变动率平方',
    '过去一月成交量变动波动率',
    '过去三月成交量波动率',
    '过去一月涨跌幅波动率',
    '过去三月涨跌幅波动率',
    '月度换手率',
    '季度换手率',
    '换手率增速平方',
    '涨跌幅',
    '过去一周指数收益率',
    '资产负债率',
    '流动比率',
    '流入额[类型]机构',
    '流入量[类型]机构',
    '流入量[类型]机构增速立方',
    '预测净利润平均值变化率',
    '预测每股收益平均值变化率',
    '预测每股现金流(CPS)平均值',
    '预测每股股利(DPS)平均值',
    '预测净资产收益率(ROE)平均值',
    '近月、次近月合约之间价格变动（年化）',
    '近月、次近月合约价格对数之差',
    '近月、次近月合约累计收益率之差',
    '主力合约前1个月收益率',
    '主力合约前1个月收益率的波动率',
    '主力合约前3个月收益率的波动率',
    '主力合约前1个月交易量的波动率',
    '主力合约前3个月交易量的波动率',
    '交易量和收益率绝对值比率的1月均值',
    '主力合约交易量增速平方',
    '主力合约前1个月收益率增速平方',
    '主力合约前1个月交易量的波动率增速立方',
    '交易量和收益率绝对值比率的1月均值增速平方',
    '净资产收益率增长率',
    '持买单量',
    '持卖单量',
    '持买单量增减',
    '持卖单量增减',]]
y = df[['点位']]

# 测试单个因子和y值的p值

for i in range(0, x.shape[1], 1):
    a = x.iloc[:, i]
    # iloc[行起始:行结束,列起始:列结束]
    # 对数据进行位置索引，从而在数据表中提取出相应的数据。
    results = sm.OLS(y, a, hasconst=False).fit()
    print(results.pvalues)