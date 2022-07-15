import os

from show_user import show_user


def show_info(user_info):
    # 显示信息
    user_list = []
    # 判断文件是否存在，如果存在则打开文件，读取信息
    if os.path.exists(user_info):
        with open(user_info, 'r', encoding='utf-8') as f:
            users = f.readlines()
            # 判断读取到的联系人信息是否为空
            if users:
                for i in users:
                    user_list.append(eval(i))
                # 展示联系人信息
                show_user(user_list)
            else:
                print('联系人文件中还没有录入联系人信息！')
    else:
        print('联系人文件不存在！')
        return