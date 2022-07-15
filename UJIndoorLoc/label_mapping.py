import numpy as np
import pandas as pd


def label_mapping_wDict(label, mode=None):
    """将Nx2维的label映射到其实际类别, 根据递归传递映射字典"""
    label0 = label[:, 0].tolist()
    label1 = label[:, 1].tolist()
    tmp = list(map(lambda a, b: str(int(a))+str(int(b)), label0, label1))
    # 去除标签列表中的重复元素
    unique_tmp = list(set(tmp))
    # print(len(unique_tmp))
    if not mode:
        # 构建非重复元素的映射字典，将标签字符串映射到类别int
        map_dict = {}
        overlap = []  # 保存已经写入映射字典的键

        flage = 0  # 类别int指示位，构建映射字典的值
        for i in tmp:
            if i not in overlap:
                tmp_dict = {i: flage}
                map_dict.update(tmp_dict)
                overlap.append(i)
                flage += 1
            if flage == len(unique_tmp):
                break

        # print(map_dict)
        # 替换原标签列表中的字符串为对应的类别int
        new_label = list(map(lambda a: map_dict.get(a), tmp))
    else:
        traindata = pd.read_csv('trainingData.csv').to_numpy()
        map_dict = label_mapping_wDict(traindata[:, 522:524])[1]
        # map_dict = label_mapping()[1]
        new_label = list(map(lambda a: map_dict.get(a), tmp))
    return np.array(new_label), map_dict


def label_mapping(label):
    """将Nx2维的label映射到其实
    际类别"""
    label0 = label[:, 0].tolist()
    label1 = label[:, 1].tolist()
    tmp = list(map(lambda a, b: str(int(a))+str(int(b)), label0, label1))
    # 去除标签列表中的重复元素
    unique_tmp = list(set(tmp))
    # print(len(unique_tmp))
    # 构建非重复元素的映射字典，将标签字符串映射到类别int
    map_dict = {}
    overlap = []  # 保存已经写入映射字典的键

    flage = 0  # 类别int指示位，构建映射字典的值
    for i in tmp:
        if i not in overlap:
            tmp_dict = {i: flage}
            map_dict.update(tmp_dict)
            overlap.append(i)
            flage += 1
            if flage == len(unique_tmp):
                break

    # print(map_dict)
    # 替换原标签列表中的字符串为对应的类别int
    new_label = list(map(lambda a: map_dict.get(a), tmp))
    return np.array(new_label)


if __name__ == '__main__':
    data = pd.read_csv('trainingData.csv')
    # data = pd.read_csv('validationData.csv')
    floor = data['FLOOR'].to_numpy()
    building = data['BUILDINGID'].to_numpy()
    label = np.zeros([len(floor), 2])
    label[:, 0] = floor
    label[:, 1] = building
    # label = label_mapping_wDict(label, mode='test')[0]
    label = label_mapping(label)
    print(list(label))