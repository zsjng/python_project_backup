import os

from show_info import show_info


def change_info(user_info):
    # 修改联系人信息
    while True:
        # 展示联系人信息
        show_info(user_info)
        # 如果user.txt存在，打开文件，读出所有信息
        if os.path.exists(user_info):
            with open(user_info, 'r', encoding='utf-8') as f:
                user_old = f.readlines()
        else:
            print('联系人文件信息不存在！')
            return
        # 创建一个新的文件，读取之前文件中的每一个联系人信息
        user_name = input('请输入要修改的联系人名字：')
        flag = False  # 标记是否在联系人信息表中找到要修改的联系人信息
        with open(user_info, 'w', encoding='utf-8') as f:
            for i in user_old:
                d = dict(eval(i))
                # 如果在原文件的某一行找到要修改的联系人名字，则对其进行修改
                if d['name'] == user_name:
                    print('找到联系人信息，可以修改相关信息')
                    while True:
                        try:
                            d['phone_num'] = input('请输入手机号:')
                            d['age'] = input('请输入年龄:')
                            d['qq_num'] = input('请输入qq号:')
                            break
                        except:
                            print('您的输入有误，请重新输入!!!')
                    f.write(str(d) + '\n')
                    flag = True
                else:
                    f.write(str(d) + '\n')
        if flag:
            print('修改成功!!!')
        else:
            print('没有找到要修改联系人的phone_num!!!')
        # 询问是否要接着修改
        yes_or_no = input('是否修改其他联系人信息？y/n\n')
        if yes_or_no == 'y' or yes_or_no == 'Y':
            continue
        else:
            break
