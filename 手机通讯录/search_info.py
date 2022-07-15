import os

from show_user import show_user


def search_info(user_info):
    # 搜索功能
    while True:
        user_query = []
        phone_num = ''
        name = ''
        # 判断文件是否存在
        if os.path.exists(user_info):
            # 输入查询标号
            mode = input('按手机号查找请按1，按姓名查找请按2：')
            if mode == '1':
                phone_num = input('请输入要查找的联系人手机号：')
            elif mode == '2':
                name = input('请输入要查找联系人的姓名:')
            else:
                print('输入错误，请重新输入!')
                continue
            # 如果文件存在，并且用户输入了正确的查询标号，则打开文件
            with open(user_info, 'r', encoding='utf-8') as f:
                users = f.readlines()
            # 将读取到的信息转换为字典类型
            for i in users:
                d = dict(eval(i))
                if phone_num != '':
                    if phone_num == d['phone_num']:
                        user_query.append(d)
                elif name != '':
                    if name == d['name']:
                        user_query.append(d)
            # 显示查询结果
            show_user(user_query)
            # 是否查询其他联系人信息
            yes_or_no = input('还要查询其他联系人吗？y/n\n')
            if yes_or_no == 'y' or yes_or_no == 'y':
                continue
            else:
                break
        else:
            print('联系人信息文件不存在！')
            return