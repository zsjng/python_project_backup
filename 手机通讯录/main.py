import os

import show_user
from change_info import change_info
from del_info import del_info
from insert_save import insert_info
from search_info import search_info
from show_info import show_info
from sort_info import sort_info
from total_info import total_info

user_info = 'user.txt'


def main():
    while True:
        menu()
        # 显示菜单
        choose = int(input('请输入功能编号：'))
        # 选择编号
        if choose in [1, 2, 3, 4, 5, 6, 7, 0]:
            # 如果选择的编号在0到7内
            if choose == 0:
                # 如果选择0
                print("确定要退出吗？y/n")
                yes_or_no = input()
                # 是否退出?
                if yes_or_no == 'y' or yes_or_no == 'Y':
                    print('欢迎下次使用')
                    break
                else:
                    continue
                # 对应7菜单
            elif choose == 1:
                insert_info(user_info)
            elif choose == 2:
                search_info(user_info)
            elif choose == 3:
                del_info(user_info)
            elif choose == 4:
                change_info(user_info)
            elif choose == 5:
                sort_info(user_info)
            elif choose == 6:
                total_info(user_info)
            elif choose == 7:
                show_info(user_info)
        else:
            print('输入错误,请重新输入对应号码')


def menu():
    print('手机通讯录软件')
    print('*' * 40)
    print('功能菜单')
    print('1. 录入联系人信息')
    print('2. 查找联系人信息')
    print('3. 删除联系人信息')
    print('4. 修改联系人信息')
    print('5. 联系人相关信息排序')
    print('6. 统计联系人总人数')
    print('7. 显示所有联系人信息')
    print('0. 退出系统')
    print('*' * 40)


if __name__ == '__main__':
    main()
