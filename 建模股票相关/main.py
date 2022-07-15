for i in range(0, x.shape[1], 1):
    c = x.iloc[:, i]
    results = sm.OLS(y2['收盘价点位'], c, hasconst=False).fit()
    d = pd.DataFrame()
    d = pd.concat([d, x.iloc[:, i]])
    d['diff'] = d[0] - d[0].shift(1)
    d = d.fillna(0)
    d['positive'] = d['diff'].apply(lambda x: 0 if x < 0 else 1)
    if (results.params[0] > 0):
        d['equal'] = np.where(d['positive'] == y2['correct'], 1, 0)
        d['res'] = d['positive'].apply(lambda x: (-1) if x != 1 else 1)
    else:
        d['equal'] = np.where(d['positive'] == y2['correct'], 0, 1)
        d['res'] = d['positive'].apply(lambda x: (-1) if x == 1 else 1)
    d['profit'] = d['res'] * open_point['diff'] * 300 * 15
    d['net_value'] = (d['profit'].cumsum() + 10000000) / 10000000
    d['retracement'] = d['net_value'] / d['net_value'].cummax() - 1
    profit2 = sum(d['profit']) / 10000000
    profit_pos2 = 1 if profit2 > 0 else -1
    profit_year2 = (pow((profit2 * profit_pos2) + 1, 1.0 / 4.5) - 1) * profit_pos2
    profit_year2 = round(profit_year2, 4)
    win2 = (sum(d['equal']) - 1) / len(d['equal'] - 1)
    max_retracement2 = d['retracement'].min()
    win_rate2 = win_rate2.append(pd.DataFrame(
        {'win_rate': win2, 'profit': profit2, 'profit_year': profit_year2, 'max_retracement': max_retracement2},
        index=[0]))
    test2 = pd.concat([test2, results.params], axis=0)

    win_rate2.index = test2.index
    test2['win_rate'] = win_rate2['win_rate']
    test2['profit'] = win_rate2['profit']
    test2['profit_year'] = win_rate2['profit_year']
    test2['max_ratracement'] = win_rate2['max_retracement']