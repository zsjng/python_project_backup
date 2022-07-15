import os

from show_info import show_info


def del_info(user_info):
    # 删除功能
    while True:
        user_name = input('请输入要删除联系人的姓名:')
        if user_name != '':
            if os.path.exists(user_info):
                with open(user_info, 'r', encoding='utf-8') as f:
                    user_old = f.readlines()
            else:
                user_old = []
            flag = False  # 标记是否删除
            if user_old:
                with open(user_info, 'w', encoding='utf-8') as f:
                    d = {}
                    for i in user_old:
                        d = dict(eval(i))  # 将字符串转换为字典
                        if d['name'] != user_name:
                            f.write(str(d) + '\n')
                        else:
                            flag = True
                    if flag:
                        print(f'姓名为{user_name}的联系人信息已经删除')
                    else:
                        print(f'联系人表中没有找到姓名为{user_name}的联系人')
            else:
                print('联系人表中无任何联系人信息')
                break
            show_info(user_info)  # 重新显示所有联系人信息
            yes_or_no = input('是否继续删除？y/n\n')
            if yes_or_no == 'y' or yes_or_no == 'Y':
                continue
            else:
                break
