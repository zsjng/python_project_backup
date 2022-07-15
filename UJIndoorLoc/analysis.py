import matplotlib.pyplot as plt
import pandas as pd


def label_analysis(data):
    # 分析
    floor = data['FLOOR']
    #
    building = data['BUILDINGID']

    # print(len(data['FLOOR']))
    # print(data['BUILDINGID'])

    building_floor = [[], [], []]
    for i in range(len(building)):
        have_floor = floor[i]
        print(have_floor)
        if have_floor not in building_floor[building[i]]:
            building_floor[building[i]].append(have_floor)
    # print(building_floor)

    res = []
    for l in building_floor:
        res.append(len(l))
    # print(res)

    x_len = len(building_floor)
    x = [i for i in range(x_len)]
    plt.figure()
    plt.bar(x, res)
    plt.xticks(x, labels=['BUILDING 0', 'BUILDING 1', 'BUILDING 2'])
    plt.ylabel('Floor')
    plt.title('Data Analysis')
    plt.show()


def feats_analysis(data):
    floor = data['FLOOR']
    building = data['BUILDINGID']
    # print(len(data['FLOOR']))
    # print(data['BUILDINGID'])

    building_floor = [[], [], []]
    for i in range(len(building)):
        have_floor = floor[i]
        if have_floor not in building_floor[building[i]]:
            building_floor[building[i]].append(have_floor)
    # print(building_floor)

    res = []
    for l in building_floor:
        res.append(len(l))
    # print(res)

    x_len = len(building_floor)
    x = [i for i in range(x_len)]
    plt.figure()
    plt.bar(x, res)
    plt.xticks(x, labels=['BUILDING 0', 'BUILDING 1', 'BUILDING 2'])
    plt.ylabel('Floor')
    plt.title('Data Analysis')
    plt.show()


if __name__ == '__main__':
    data_train = pd.read_csv('trainingData.csv')
    data_test = pd.read_csv('validationData.csv')

    label_analysis(data_train)

    # 结果与训练集相同
    label_analysis(data_test)

