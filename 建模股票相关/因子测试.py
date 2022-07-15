# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:18:16 2021

@author: glqh
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 导入数据

excelFile = r'./hs300因子优化1710-2203模型.xlsx'
df = pd.DataFrame(pd.read_excel(excelFile, sheet_name='建模数据'))
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date')
# new_df=df[df.index.day==15]

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
    '持卖单量增减']]

# y值取收益率，直接比较收益率正负
y = df[['主力合约收益率']].copy()
y['correct'] = y['主力合约收益率'].apply(lambda x: 0 if x < 0 else 1)  # 收益率判断正负
# 单纯的判断正负
# y=y.reset_index(drop=True)

# y2值取收盘价点位，需要计算收盘价两日之差
y2 = df[['主力合约收盘价']].copy()
y2['收盘价点位'] = df[['收盘价点位']].copy()
y2['diff'] = y2['主力合约收盘价'] - y2['主力合约收盘价'].shift(1)
# 计算 与上一行的差值 shift(1)就是下移一行,也就是用本行 - 上一行
y2['correct'] = y2['diff'].apply(lambda x: 0 if x < 0 else 1)  # 收盘价变动方向判断正负
# 判断收盘价是涨还是跌

# 计算开盘收盘价格差
open_point = df[['主力合约收盘价']].copy()

open_point['收盘价点位'] = df[['主力合约收盘价']].copy()
open_point['开盘价点位'] = df[['主力合约收盘价']].copy().shift(1)
open_point['diff'] = open_point['收盘价点位'] - open_point['开盘价点位']
# print(open_point['diff'])

test = pd.DataFrame()

win_rate = pd.DataFrame()

for i in range(0, x.shape[1], 1):
    # 从0 到 75 X有75项
    a = x.iloc[:, i]
    # a = x的第i项
    results = sm.OLS(y['主力合约收益率'], a, hasconst=False).fit()
    # statsmodels.OLS 的输入有 (endog, exog, missing, hasconst) 四个，我们现在只考虑前两个。
    # 第一个输入 endog 是回归中的反应变量(也称因变量)，是上面模型中的y['主力合约收益率'], 输入是一个长度为 k 的 array。
    # 第二个输入 exog 则是回归变量(也称自变量)的值，即模型中的x1(t),…,xn(t)。
    # 但是要注意，statsmodels.OLS 不会假设回归模型有常数项，所以我们应该假设模型是 见网页
    b = pd.DataFrame()

    b = pd.concat([b, x.iloc[:, i]])
    # 沿着指定的轴将多个dataframe或者series拼接到一起
    b['diff'] = b[0] - b[0].shift(1)  # 因子变动方向
    # 作差
    b = b.fillna(0)
    # 用0来填充缺失的
    b['positive'] = b['diff'].apply(lambda x: 0 if x < 0 else 1)  # 因子增加1减少0
    # 正向相关
    # print(results[0])

    if (results.params[0] > 0):
        # 皮尔森系数大于0
        b['equal'] = np.where(b['positive'] == y['correct'], 1, 0)  # 对比因子和收益率的方向是否相同
        # 判断正负是否相关
        b['res'] = b['positive'].apply(lambda x: (-1) if x != 1 else 1)  # 因子减少记为-1，增加记为1，用于计算绩效

    else:
        b['equal'] = np.where(b['positive'] == y['correct'], 0, 1)

        b['res'] = b['positive'].apply(lambda x: (-1) if x == 1 else 1)

    b['profit'] = b['res'] * open_point['diff'] * 300 * 15  # 计算绩效

    b['net_value'] = (b['profit'].cumsum() + 10000000) / 10000000
    # 样本值累加求和

    b['retracement'] = b['net_value'] / b['net_value'].cummax() - 1
    # 回撤率
    # print(b['profit'].cumsum())

    profit = sum(b['profit']) / 10000000
    print(profit)
    profit_pos = 1 if profit > 0 else -1

    profit_year = (pow((profit * profit_pos) + 1, 1.0 / 4.5) - 1) * profit_pos
    profit_year = round(profit_year, 4)
    # 年利润
    # print(profit_year)
    # 四舍五入保留小数点后4位

    win = (sum(b['equal']) - 1) / len(b['equal'] - 1)
    # 因子和收益率的方向
    max_retracement = b['retracement'].min()

    # 最大回撤率
    win_rate = win_rate.append(pd.DataFrame(
        {'win_rate': win, 'profit': profit, 'profit_year': profit, 'max_retracement': max_retracement}, index=[0]))
    #     win_rate  profit  profit_year  max_retracement
    # 0   0.493126     NaN          NaN        -0.701641

    test = pd.concat([test, results.params], axis=0)
    # 讲两个数据以横向拼接,axis = 1则是竖向
    # print(test)
    # 计算每一个 X 对于 y的影响因子系数
    win_rate.index = test.index
    test['win_rate'] = win_rate['win_rate']
    test['profit'] = win_rate['profit']
    test['profit_year'] = win_rate['profit_year']
    test['max_ratracement'] = win_rate['max_retracement']
    print(win_rate)
    win_rate.to_excel('./abc.xlsx')
    # test2 = pd.DataFrame()
    #
    # win_rate2 = pd.DataFrame()
    # 定义两个空白的数据类型
    # # for i in range(0, x.shape[1], 1):
    # #     # 0 到 75 项
    # #     c = x.iloc[:, i]
    # #     # 选取 其中的第i列
    # #     results = sm.OLS(y2['收盘价点位'], c, hasconst=False).fit()
    # #     # 通过OLS线性回归计算
    # #     d = pd.DataFrame()
    # #
    # #     d = pd.concat([d, x.iloc[:, i]])
    # #     # 将对应数据拼接,默认axis=0
    # #     d['diff'] = d[0] - d[0].shift(1)
    # #     # 计算与前一日的差值
    # #     d = d.fillna(0)
    # #     # 没有的用0来填充
    # #     d['positive'] = d['diff'].apply(lambda x: 0 if x < 0 else 1)
    # #     # 判断正负,如果小于0就是负,大于0就是正
    # #     if (results.params[0] > 0):
    # #         # 皮尔森系数
    # #         d['equal'] = np.where(d['positive'] == y2['correct'], 1, 0)
    # #         #
    # #         d['res'] = d['positive'].apply(lambda x: (-1) if x != 1 else 1)
    # #
    # #     else:
    # #         d['equal'] = np.where(d['positive'] == y2['correct'], 0, 1)
    # #         d['res'] = d['positive'].apply(lambda x: (-1) if x == 1 else 1)
    # #     d['profit'] = d['res'] * open_point['diff'] * 300 * 15
    # #     d['net_value'] = (d['profit'].cumsum() + 10000000) / 10000000
    # #     d['retracement'] = d['net_value'] / d['net_value'].cummax() - 1
    # #     profit2 = sum(d['profit']) / 10000000
    # #     profit_pos2 = 1 if profit2 > 0 else -1
    # #     profit_year2 = (pow((profit2 * profit_pos2) + 1, 1.0 / 4.5) - 1) * profit_pos2
    # #     profit_year2 = round(profit_year2, 4)
    # #     win2 = (sum(d['equal']) - 1) / len(d['equal'] - 1)
    # #     max_retracement2 = d['retracement'].min()
    # #     win_rate2 = win_rate2.append(pd.DataFrame(
    # #         {'win_rate': win2, 'profit': profit2, 'profit_year': profit_year2, 'max_retracement': max_retracement2},
    # #         index=[0]))
    # #     test2 = pd.concat([test2, results.params], axis=0)
    # #
    # #     win_rate2.index = test2.index
    # #     test2['win_rate'] = win_rate2['win_rate']
    # #     test2['profit'] = win_rate2['profit']
    # #     test2['profit_year'] = win_rate2['profit_year']
    # #     test2['max_ratracement'] = win_rate2['max_retracement']
