def show_user(info_list):
    # 传过来列表
    if len(info_list) == 0:
        print('没有查找到该联系人信息！')
        return
    # 定义标题显示格式
    format_title = '{:^12}\t{:^8}\t{:^8}\t{:^12}\t'
    print(format_title.format('手机号', '姓名', '年龄', 'QQ号'))
    # 定义内容的显示格式
    format_data = '{:^12}\t{:^8}\t{:^8}\t{:^12}\t'
    for i in info_list:
        print(format_data.format(i['phone_num'], i['name'],
                                 i['age'], i['qq_num']))