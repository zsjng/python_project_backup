def insert_info(user_info):
    # 录入和存储功能
    user_list = []
    while True:
        phone_num = int(input('请输入手机号码:'))
        # 判定手机号是否为整数
        if not phone_num:
            break
        name = input('请输入姓名:')
        # 如果姓名为空就报错
        if not name:
            break
        try:
            age = int(input('请输入年龄：'))
            qq_num = int(input('请输入QQ号：'))
        except:
            print('输入无效请重新输入整数')
            continue
        # 将录入的联系人信息保存到字典中
        user = {'phone_num': phone_num, 'name': name, 'age': age, 'qq_num': qq_num}
        # 将联系人信息添加到列表中
        user_list.append(user)
        yes_or_no = input('是否继续添加？y/n\n')
        if yes_or_no == 'y' or yes_or_no == 'Y':
            continue
        else:
            break
    # 保存联系人信息到文件中
    save_info(user_list, user_info)
    print('联系人信息录入完毕!!!')


def save_info(info_list, user_info):
    # 保存功能
    user_txt = open(user_info, 'a', encoding='utf-8')
    # 打开user_txt ,在文件末尾写入
    for i in info_list:
        # 每1个人存放一行
        user_txt.write(str(i) + '\n')
    user_txt.close()
    # 关闭user_txt
