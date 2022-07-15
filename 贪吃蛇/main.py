import pygame
import random

# 初始化游戏界面
pygame.init()
map_width = 640
map_height = 480
# 定义地图网格,row行 col列
map_row = 24
map_col = 32
# 得分和加速
count = 0
fps_count = 0
should_rank = True
# 初始化游戏窗口,大小由size控制:宽,高
size = (map_width, map_height)
window = pygame.display.set_mode(size)
fps = 10
# 设置游戏名称 左上角现实的标题
pygame.display.set_caption('贪吃蛇小游戏')


# 定义坐标
class Point:
    def __init__(self, row=0, col=0):
        self.row = row
        self.col = col

    def copy_point(self):
        return Point(row=self.row, col=self.col)


# 地图元素定义
# 蛇头坐标 在地图中间,颜色为红色
snake_head = Point(row=int((map_row / 2) * random.randint(1, 10) / 10),
                   col=int((map_col / 2) * random.randint(1, 10) / 10))
snake_head_color = (255, 0, 0)
# 上 下 左 右,初始化蛇头朝向
snake_head_direct = 'up'
# 蛇 身体
snake_body = [
    Point(row=snake_head.row + 1, col=snake_head.col),
    Point(row=snake_head.row + 2, col=snake_head.col),
    Point(row=snake_head.row + 3, col=snake_head.col),
]
snake_body_color = (180, 180, 180)


# 生成奖励
def generator_reward():
    while True:
        pos = Point(row=random.randint(0, map_row - 1), col=random.randint(0, map_col - 1))

        # 是否碰上
        is_peng = False
        if snake_head.row == pos.row and snake_head.col == pos.col:
            is_peng = True
        # 判断身体
        for i in snake_body:
            if i.row == pos.row and i.col == pos.col:
                is_peng = True
                break

        if not is_peng:
            break
    return pos


# 奖励坐标 初始 随机 ,绿色
reward = generator_reward()
reward_color = (0, 255, 0)


def rect(point, color):
    # 网格宽高
    grid_width = map_width / map_col
    grid_height = map_height / map_row
    # 定义宽/高的实际坐标而不是网格位置
    left = point.col * grid_width
    top = point.row * grid_height
    # 画矩形,
    pygame.draw.rect(
        window, color, (left, top, grid_width, grid_height)
    )


# 游戏本体
clock = pygame.time.Clock()
# 定义游戏时间
bool_value = True
# 暂停
is_pause = False
while bool_value:
    events = pygame.event.get()

    # 游戏的各种事件
    for event in events:

        # print(event)
        # 如果事件类型为 退出!(点击右上角的X)
        if event.type == pygame.QUIT:
            should_rank = False
            bool_value = False
        elif event.type == pygame.KEYDOWN:
            if event.key == 1073741906:
                if snake_head_direct == 'left' or snake_head_direct == 'right':
                    snake_head_direct = 'up'
            elif event.key == 1073741905:
                if snake_head_direct == 'left' or snake_head_direct == 'right':
                    snake_head_direct = 'down'
            elif event.key == 1073741904:
                if snake_head_direct == 'up' or snake_head_direct == 'down':
                    snake_head_direct = 'left'
            elif event.key == 1073741903:
                if snake_head_direct == 'up' or snake_head_direct == 'down':
                    snake_head_direct = 'right'
            elif event.key == 32:
                is_pause = not is_pause
    # 暂停功能,非常简单

    if is_pause:
        continue
    # 判断蛇头是否和奖励重合
    get_reward = (snake_head.col == reward.col and snake_head.row == reward.row)

    # 如果获得奖励就重新生成新的奖励
    if get_reward:
        count += 10
        print(f'当前得分为{count}，加油！')
        fps_count += 1
        reward = generator_reward()
    snake_body.insert(0, snake_head.copy_point())
    if not get_reward:
        snake_body.pop()

    # 定义动作
    if snake_head_direct == 'up':
        snake_head.row -= 1
    elif snake_head_direct == 'down':
        snake_head.row += 1
    elif snake_head_direct == 'left':
        snake_head.col -= 1
    elif snake_head_direct == 'right':
        snake_head.col += 1

    # 判断撞墙
    is_dead = False
    if map_col < snake_head.col or snake_head.col < 0 or map_row < snake_head.row or snake_head.row < 0:
        is_dead = True

    # 碰到身子 头坐标和身体坐标重复.
    for i in snake_body:
        if snake_head.col == i.col and snake_head.row == i.row:
            is_dead = True
            break
    if is_dead:
        bool_value = False
    # 初始化游戏,渲染界面(渲染对象 window ,颜色 白色(255,255,255)
    # 坐标(左上0,0 右下 width,height)这里和一开始初始化尺寸相同
    pygame.draw.rect(window, (255, 255, 255), (0, 0, map_width, map_height))
    # 在地图上绘制出蛇头和奖励
    # x,y画网格
    for x in range(0, map_width, 20):
        pygame.draw.line(window, (0, 0, 0), (x, 0), (x, map_height), 1)
    for y in range(0, map_height, 20):
        pygame.draw.line(window, (0, 0, 0), (0, y), (map_width, y), 1)
    rect(snake_head, snake_head_color)
    rect(reward, reward_color)
    for i in snake_body:
        rect(i, snake_body_color)

    pygame.display.flip()
    # 将控制权交给系统
    # 保存数据
    clock.tick(fps + fps_count)
    # 设置游戏帧数,fps


def ranking_count(should_rank):
    if should_rank:
        print('游戏结束,game over')
        print(f'您的得分为{count}')
        name = input('请输入姓名:\n')
        ranking = f'{name},{count}!'
        with open('rank.txt', 'a+', encoding='utf-8') as f:
            f.write(ranking)
    else:
        pass


def rank(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        rank_list = f.readline()
    list_1 = rank_list.split('!')
    rank = []
    for i in list_1:
        rank.append(i.split(','))
    rank = rank[:-1]
    new_rank = sorted(rank, key=lambda x: (x[1], x[0]), reverse=True)
    print('------高分榜------')
    for j in new_rank[0:5]:
        # 只显示前五名最高分

        print(j, sep='\n')


ranking_count(should_rank)
rank('rank.txt')
