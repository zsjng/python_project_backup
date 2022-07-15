import os


def total_info(user_info):
    # 判断文件是否存在，如果存在则打开文件，读取信息
    if os.path.exists(user_info):
        with open(user_info, 'r', encoding='utf-8') as f:
            users = f.readlines()
            # 判断读取到的联系人信息是否为空
            if users:
                print('一共有{}名联系人'.format(len(users)))
            else:
                print('还没有录入联系人！')
    else:
        print('user.txt不存在！')
        return
