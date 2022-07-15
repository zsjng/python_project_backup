def draw(h, l):
    a, b, c, d = "＋", "- ", "丨", "  "
    hang = 4 * b + a
    lie = 4 * d + c
    for i in range(h):
        print(a + hang * l)
        print(c + lie * l)
        print(c + lie * l)
        print(c + lie * l)
        # print(c + ch*l)
    print(a + hang * l)


def myFunc():
    a = input("输入行和列(用逗号隔开)：")
    h, l = a[0], a[-1]
    draw(int(h), int(l))


myFunc()
