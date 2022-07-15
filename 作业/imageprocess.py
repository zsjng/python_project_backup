import os
import sys

from PIL import Image


def change_logo():
    if os.path.exists('./pic_out/'):
        pass
    else:
        os.mkdir('./pic_out/')
    im_logo = Image.open('logo.png')
    # 将logo缩小为 160 * 160

    new_logo = im_logo.resize((160, 160))
    # 将logo转化为四通道rgba文件
    new_logo = new_logo.convert('RGBA')
    x, y = new_logo.size
    for i in range(x):
        for j in range(y):
            color = new_logo.getpixel((i, j))
            color = color[:-1] + (127,)
            new_logo.putpixel((i, j), color)
    new_logo.save('new_logo.png')


# 将pic的所有jpg文件转化为png文件
def change_file():
    path = os.path.abspath(sys.path[0])
    for filename in os.listdir(f'{path}/pic/'):
        print(filename)
        new_filename = filename[:filename.find('.')] + '.png'
        im = Image.open(f'{path}/pic/{filename}')
        im.save(path + '/pic_out/' + new_filename)


def merge(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im


def add_pic(path):
    #
    image = Image.open(f'./pic_out/{path}')
    im2 = Image.open('new_logo.png')
    image.paste(im2, (0, 0), im2)
    image.show()
    if os.path.exists('./pic_add'):
        image.save('./pic_add/' + f'{path}')
    else:
        os.mkdir('./pic_add')
        image.save('./pic_add/' + f'{path}')


def main():
    for file in os.listdir('./pic_out/'):
        add_pic(file)


if __name__ == '__main__':
    change_logo()
    change_file()
    main()
