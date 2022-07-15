import os
from show_info import show_info
from show_user import show_user


def sort_info(user_info):
    # 排序功能
    show_info(user_info)
    # 判断文件是否存在，如果存在则打开文件，读取信息
    if os.path.exists(user_info):
        with open(user_info, 'r', encoding='utf-8') as f:
            users_list = f.readlines()
        users_new = []
        # 判断读取到的联系人信息是否为空
        if users_list:
            # 将所有的i加入到users_new中
            for i in users_list:
                d = dict(eval(i))
                users_new.append(d)
            # 选择升序or降序
            up_or_down = input('请选择(0为升序，1为降序):')
            up_or_down_bool = False
            if up_or_down == '0':
                up_or_down_bool = False
            elif up_or_down == '1':
                up_or_down_bool = True
            else:
                print('您的输入有误，请重新输入')
                sort_info()
            # 选择按照什么方式排序
            mode = input('请选择排序方式（1.按年龄大小排序 2.按QQ号大小排序 3.按手机号大小):')
            # 通讯录实在没啥好排序的..也就是年龄了
            if mode == '1':
                users_new.sort(key=lambda x: int(x['age']), reverse=up_or_down_bool)
                # lambda 这里相当于对x的age列尽进行排序,升降序由你自己选择,
                # 下两个依次对qq 手机号排列,虽然不知道手机号排列有啥意义..
            elif mode == '2':
                users_new.sort(key=lambda x: int(x['qq_num']), reverse=up_or_down_bool)
            elif mode == '3':
                users_new.sort(key=lambda x: int(x['phone_num']), reverse=up_or_down_bool)
            else:
                print('您的输入有误，请重新输入！！！')
                sort_info()
            # 排序后进行输出
            show_user(users_new)
        else:
            print('联系人文件中还没有录入联系人信息！')
    else:
        print('联系人文件不存在！')
        return
