import re
import json
import time
import base64
from datetime import datetime
from pathlib import Path, WindowsPath
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from moviepy.editor import *
from pydub import AudioSegment
from pydub.silence import split_on_silence
from queue import Queue

from utils import get_baidu_result, get_baidu_ocr_token, get_baidu_ocr_result

from cv2 import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

with open('settings.json') as f:
    settings = json.loads(f.read())


VIDEO = 0
AUDIO = 1


def frame_to_base64(frame):
    data = cv2.imencode('.jpg', frame)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def video2audio(video_path, audio_path):
    """将视频文件中音频提取出来
       音频编码要求：采样率 16000、8000（仅支持普通话模型），16 bit 位深，单声道（音频格式查看及转换）
    """
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, ffmpeg_params=['-ar', '16000', '-ac', '1'])


class ImageRecognition(QThread):
    """百度OCR识别"""
    def __init__(self, insert_text_signal, insert_result_signal, insert_content_signal, start_btn_signal, task_queue):
        super(ImageRecognition, self).__init__()
        self.task_queue = task_queue
        # self.video_path = video_path
        self.insert_text_signal = insert_text_signal
        self.insert_result_signal = insert_result_signal
        self.insert_content_signal = insert_content_signal
        self.start_btn_signal = start_btn_signal

    def get_video_info(self):
        """获取视频文件信息"""
        v = cv2.VideoCapture(str(self.video_path))
        num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = v.get(cv2.CAP_PROP_FPS)
        height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_info = {
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
        }
        return video_info

    def get_word_vector(self, s1, s2):
        """
        Args：
            s1: 字符串1
            s2: 字符串2
        return:
            返回字符串切分后的向量
        """
        # 字符串中文按字分，英文按单词，数字按空格
        regEx = re.compile('[\\W]*')
        res = re.compile(r"([\u4e00-\u9fa5])")
        p1 = regEx.split(s1.lower())
        str1_list = []
        for str in p1:
            if res.split(str) == None:
                str1_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str1_list.append(ch)
        # print(str1_list)
        p2 = regEx.split(s2.lower())
        str2_list = []
        for str in p2:
            if res.split(str) == None:
                str2_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str2_list.append(ch)
        # print(str2_list)
        list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
        list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符
        # 列出所有的词,取并集
        key_word = list(set(list_word1 + list_word2))
        # 给定形状和类型的用0填充的矩阵存储向量
        word_vector1 = np.zeros(len(key_word))
        word_vector2 = np.zeros(len(key_word))
        # 计算词频
        # 依次确定向量的每个位置的值
        for i in range(len(key_word)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(list_word1)):
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1
            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1

        # 输出向量
        return word_vector1, word_vector2

    def cos_dist(self, vec1, vec2):
        """
        Args:
            vec1: 向量1
            vec2: 向量2
        return:
            返回两个向量的余弦相似度
        """
        dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return dist1

    def calu_dist(self, result, last_result):
        if last_result is None:
            return 0
        s1 = "".join(result)
        s2 = "".join(last_result)
        word_vector1, word_vector2 = self.get_word_vector(s1, s2)
        dist = self.cos_dist(word_vector1, word_vector2)
        return dist

    def tailor_video(self, video_info):
        """对视频帧进行裁剪
        Args:
            frame_frequency 截取帧间隔
        """
        API_KEY = settings.get('video').get('API_KEY')
        SECRET_KEY = settings.get('video').get('SECRET_KEY')
        height = video_info.get('height')
        frame_frequency = settings.get('video', {}).get('frame_frequency', 12)
        h1, h2 = settings.get('video', {}).get('h1', 0.8), settings.get('video', {}).get('h2', 1.0)
        h1, h2 = int(height * h1), int(height * h2)
        cap = cv2.VideoCapture(str(self.video_path))
        idx = 0
        last_result = None
        while True:
            idx += 1
            ret = cap.grab()
            if not ret:
                break
            if idx % frame_frequency == 1:
                ret, frame = cap.retrieve()
                if frame is None:
                    break
                cropped = frame[h1:h2, :]  # 按字幕位置裁剪
                # blur_img = cv2.GaussianBlur(cropped, (0, 0), 5)
                # img_usm = cv2.addWeighted(cropped, 1.5, blur_img, -0.5, 0)  # usm锐化
                img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # 灰度
                img_bilatera = cv2.bilateralFilter(img_gray, 0, 40, 20)  # 高斯双边滤波
                img = cv2.bitwise_not(img_bilatera)  # 取反
                score = cv2.Laplacian(img, cv2.CV_64F).var()  # 清晰度
                if score >= 700:
                    image_base64 = frame_to_base64(img)
                    token = get_baidu_ocr_token(API_KEY, SECRET_KEY)
                    result = get_baidu_ocr_result(token, image_base64)
                    # binary_data = base64.b64decode(image_base64)
                    # img_data = BytesIO(binary_data)
                    # file = Image.open(img_data)
                    # file.show()
                    # result = reader.readtext(img, detail=0)
                    if result:
                        dist = self.calu_dist(result, last_result)
                        if dist < 0.8:
                            last_result = result
                            self.insert_result_signal.emit("".join(result))
                            self.insert_content_signal.emit("".join(last_result))
        cap.release()

    def run(self):
        self.insert_text_signal.emit('开始字幕识别, 如果视频文件过大，请等待')
        while True:
            if not self.task_queue.empty():
                self.video_path = self.task_queue.get()
                time.sleep(1)
                video = self.get_video_info()
                self.tailor_video(video)
            else:
                self.insert_text_signal.emit('结束字幕识别')
                self.start_btn_signal.emit(False)
                break


class SpeechRecognition(QThread):
    """通过百度API进行语音识别"""
    def __init__(self, insert_text_signal, insert_result_signal, insert_content_signal, start_btn_signal, task_queue, chunks_dir='./chunks', audio_file_path="out.wav"):
        super(SpeechRecognition, self).__init__()
        self.task_queue = task_queue
        self.insert_text_signal = insert_text_signal
        self.insert_result_signal = insert_result_signal
        self.insert_content_signal = insert_content_signal
        self.start_btn_signal = start_btn_signal
        self.audio_file_path = audio_file_path
        self.chunks_dir = Path(chunks_dir)

    def split_audio(self):
        """分割音频
            Args:
                silence_thresh: 默认-40      # 小于-40dBFS以下的为静默
                min_silence_len: 默认450     # 静默超过450毫秒则拆分
            return:
                chunks 音频分段
        """
        silence_thresh = settings.get('audio', {}).get('silence_thresh', -40)
        min_silence_len = settings.get('audio', {}).get('min_silence_len', 450)
        self.insert_text_signal.emit('正在拆分音频, 如果音频较长，请耐心等待')
        time.sleep(1)
        sound = AudioSegment.from_file(self.audio_file_path, format="wav")
        chunks = split_on_silence(sound, min_silence_len, silence_thresh, keep_silence=500)
        self.insert_text_signal.emit('拆分结束, 共分拆{}段:'.format(len(chunks)))
        return chunks

    def join_audio(self, chunks, joint_silence_len=1000, length_limit=60 * 1000):
        """音频合并, 提高识别效率
           百度语音识别接口限制：录音文件要求时长不超过 60 秒。
           Args:
                chunks: 音频分段
                length_limit: 默认60*1000    # 拆分后每段不得超过1分钟
                joint_silence_len: 默认1300  # 段拼接时加入1300毫秒间隔用于断句
           return:
                chunks 合并后音频分段
        """
        self.insert_text_signal.emit('正在合并过短的音频, 请耐心等待')
        silence = AudioSegment.silent(duration=joint_silence_len)
        adjust_chunks = []
        temp = AudioSegment.empty()
        for chunk in chunks:
            length = len(temp) + len(silence) + len(chunk)  # 预计合并后长度
            if length < length_limit:  # 小于1分钟，可以合并
                temp += silence + chunk
            else:  # 大于1分钟，先将之前的保存，重新开始累加
                adjust_chunks.append(temp)
                temp = chunk
        else:
            adjust_chunks.append(temp)
        return adjust_chunks

    def export_chunks(self, chunks):
        """导出音频分段"""
        total = len(chunks)
        for i in range(total):
            new = chunks[i]
            save_name = '%s_%04d.%s' % ('temp', i, 'wav')
            new_file_path = BASE_DIR / 'chunks' / save_name
            new.export(new_file_path, format='wav')
            self.insert_text_signal.emit('导出文件:{}'.format(new_file_path))

    def speech_recognition(self):
        """上传并识别语音"""
        self.insert_text_signal.emit('开始上传并识别语音')
        API_KEY = settings.get('audio').get('API_KEY')
        SECRET_KEY = settings.get('audio').get('SECRET_KEY')
        for i in self.chunks_dir.glob("**/*.wav"):
            audio_file_path = BASE_DIR / i
            result = get_baidu_result(API_KEY, SECRET_KEY, audio_file_path)
            result_data = json.loads(result)
            try:
                result_text = result_data.get('result')[0]
                self.insert_result_signal.emit(result_text)
                self.insert_content_signal.emit(result_text)
            except TypeError as e:
                time.sleep(2)

        self.insert_text_signal.emit("识别结束")

    def clean_dir(self):
        """清空输出文件夹"""
        for i in self.chunks_dir.glob("**/*.wav"):
            try:
                os.remove(i)
            except PermissionError as e:
                print("删除失败, {}".format(e))

    def run(self):
        while True:
            if not self.task_queue.empty():
                video_file_path = self.task_queue.get()
                audio_file_path = Path(BASE_DIR) / 'out.wav'
                self.insert_text_signal.emit("正在提取音频: {}".format(audio_file_path))
                video2audio(str(video_file_path), audio_file_path)
                self.insert_text_signal.emit("提取音频成功: {}".format(audio_file_path))
                self.clean_dir()
                chunks = self.split_audio()
                new_chunks = self.join_audio(chunks)
                self.export_chunks(new_chunks)
                self.speech_recognition()
            else:
                self.start_btn_signal.emit(False)
                break


class MainWindow(QMainWindow):
    insert_text_signal = pyqtSignal(str)
    insert_result_signal = pyqtSignal(str)
    insert_content_signal = pyqtSignal(str)
    start_split_audio_signal = pyqtSignal()
    start_speech_recognition_signal = pyqtSignal()
    start_btn_signal = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.status_bar = QStatusBar()
        self.init_ui()
        self.init_status_bar()
        self.mode = VIDEO
        self.current_video_path = []
        self.contents = []
        self.insert_text_signal.connect(self.insert_output)
        self.insert_content_signal.connect(self.insert_content)
        self.insert_result_signal.connect(self.insert_result)
        self.start_btn_signal.connect(self.change_btn_status)
        self.task_queue = Queue()

    def init_ui(self):
        self.setWindowTitle('视频文本检索')
        self.setWindowIcon(QIcon("app.ico"))
        self.setFixedSize(900, 620)
        self.move_center()
        self.init_tabs()

    def init_tabs(self):
        """初始化tabs选项卡"""
        # tabs = QTabWidget()
        # tabs.setStyleSheet("font-size:14px;font-family:Arial,Microsoft YaHei,黑体,宋体,sans-serif;")
        self.tab1 = QWidget()
        self.init_tab1_ui()
        # self.tab2 = QWidget()

        # tabs.addTab(self.tab1, "字幕识别")
        # tabs.addTab(self.tab2, "配置")

        # tabs.currentChanged['int'].connect(self.tab_on_click)

        # self.init_tab2_ui()
        self.setCentralWidget(self.tab1)

    def init_status_bar(self):
        """初始化状态栏"""
        self.status_bar.addPermanentWidget(QLabel(""))
        self.status_bar.showMessage('状态栏')
        self.setStatusBar(self.status_bar)

    def move_center(self):
        """窗口居中显示"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_tab1_ui(self):
        main_layout = QVBoxLayout()
        self.tab1.setLayout(main_layout)

        # 菜单选项布局
        menu = QWidget()
        menu_layout = QGridLayout()
        # menu_layout.setContentsMargins(8, 4, 8, 4)
        menu_layout.setAlignment(Qt.AlignTop)
        video_path_label = QLabel('文件路径:')
        self.video_path_input = QLineEdit('')
        video_path_button = QPushButton("选择文件")
        video_path_button.clicked.connect(self.open_file)
        cb = QComboBox()
        cb.addItems(['字幕识别', '语音识别'])
        cb.currentIndexChanged[int].connect(self.cb_on_click)
        match_label = QLabel('查询:')
        self.match_input = QLineEdit('')
        match_button = QPushButton("查询")
        # refresh_button = QPushButton("刷新")
        export_button = QPushButton("导出结果")
        export_button.clicked.connect(self.save_file)
        match_button.clicked.connect(self.match_word)
        # refresh_button.clicked.connect(self.refresh)
        menu_layout.setSpacing(14)
        menu_layout.addWidget(video_path_label, 0, 0)
        menu_layout.addWidget(self.video_path_input, 0, 1)
        menu_layout.addWidget(video_path_button, 0, 3)
        menu_layout.addWidget(cb, 0, 2)
        menu_layout.addWidget(match_label, 1, 0)
        menu_layout.addWidget(self.match_input, 1, 1)
        menu_layout.addWidget(match_button, 1, 2)
        # menu_layout.addWidget(refresh_button, 1, 3)
        menu_layout.addWidget(export_button, 0, 4)
        menu.setLayout(menu_layout)

        # 查找结果布局
        result = QWidget()
        result_layout = QHBoxLayout()
        result_layout.setAlignment(Qt.AlignTop)
        result_group = QGroupBox('识别结果', self)
        result_area = QVBoxLayout(result_group)
        self.text_result_content = QTextEdit("")
        self.text_result_content.setReadOnly(True)
        result_group.setFixedHeight(200)
        result_area.addWidget(self.text_result_content)
        result_layout.addWidget(result_group)
        result.setLayout(result_layout)

        # 输出信息布局
        output = QWidget()
        output_layout = QHBoxLayout()
        output_layout.setAlignment(Qt.AlignTop)
        top_group = QGroupBox('输出信息', self)
        output_area = QVBoxLayout(top_group)
        self.text_output_content = QTextEdit("")
        top_group.setFixedHeight(180)
        output_area.addWidget(self.text_output_content)
        output_layout.addWidget(top_group)
        output.setLayout(output_layout)

        # 底部开始按钮布局
        bottom = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignRight)
        self.start_button = QPushButton("运行")
        self.start_button.clicked.connect(self.execute)
        self.start_button.setFixedSize(80, 35)
        self.start_button.setStyleSheet("font-size:16px;")
        bottom_layout.addWidget(self.start_button)
        bottom.setLayout(bottom_layout)

        main_layout.addWidget(menu)
        main_layout.addWidget(result)
        main_layout.addWidget(output)
        main_layout.addWidget(bottom)

    def init_tab2_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        self.tab2.setLayout(main_layout)

        # 字幕配置选项布局
        video_setting = QWidget()
        video_setting_layout = QHBoxLayout()
        # video_setting_layout.setAlignment(Qt.AlignTop)
        video_setting_box = QGroupBox('字幕识别参数', self)
        video_setting_area = QGridLayout(video_setting_box)
        video_setting_area.setSpacing(14)
        intValidator = QIntValidator()
        intValidator.setRange(1, 240)
        doubleValidator = QDoubleValidator()
        label_frame_frequency = QLabel("截取帧间隔:")
        self.input_frame_frequency = QLineEdit("{}".format(settings.get('video', {}).get('frame_frequency'), 12))
        self.input_frame_frequency.setPlaceholderText("默认12")
        self.input_frame_frequency.setValidator(intValidator)
        label_h1 = QLabel("h1 高度比例:")
        self.input_h1 = QLineEdit("{}".format(settings.get('video', {}).get('h1'), 0.8))
        self.input_h1.setPlaceholderText("默认0.8")
        self.input_h1.setValidator(doubleValidator)
        label_h2 = QLabel("h2 高度比例:")
        self.input_h2 = QLineEdit("{}".format(settings.get('video', {}).get('h2'), 1))
        self.input_h2.setPlaceholderText("默认 1")
        self.input_h2.setValidator(doubleValidator)
        video_setting_area.addWidget(label_frame_frequency, 0, 0)
        video_setting_area.addWidget(self.input_frame_frequency, 0, 1)
        video_setting_area.addWidget(label_h1, 0, 3)
        video_setting_area.addWidget(self.input_h1, 0, 4)
        video_setting_area.addWidget(label_h2, 0, 5)
        video_setting_area.addWidget(self.input_h2, 0, 6)

        video_setting_box.setFixedHeight(60)
        video_setting_layout.addWidget(video_setting_box)
        video_setting.setLayout(video_setting_layout)



        # 音频配置选项布局
        audio_setting = QWidget()
        audio_setting_layout = QHBoxLayout()
        # audio_setting_layout.setAlignment(Qt.AlignTop)
        audio_setting_box = QGroupBox('语音识别参数', self)
        audio_setting_area = QGridLayout(audio_setting_box)
        intValidator = QIntValidator()

        label_API_KEY = QLabel("百度 API_KEY:")
        self.input_API_KEY = QLineEdit("{}".format(settings.get('audio', {}).get('API_KEY'), ""))

        label_SECRET_KEY = QLabel("百度 SECRET_KEY:")
        self.input_SECRET_KEY = QLineEdit("{}".format(settings.get('audio', {}).get('SECRET_KEY'), ""))

        label_silence_thresh = QLabel("静默分贝值:")
        self.input_silence_thresh = QLineEdit("{}".format(settings.get('audio', {}).get('silence_thresh'), -40))
        self.input_silence_thresh.setPlaceholderText("默认 -45")
        self.input_silence_thresh.setValidator(intValidator)

        label_min_silence_len = QLabel("静默时长:")
        self.input_min_silence_len = QLineEdit("{}".format(settings.get('audio', {}).get('min_silence_len'), 450))
        self.input_min_silence_len.setPlaceholderText("默认 450")
        self.input_min_silence_len.setValidator(intValidator)

        audio_setting_area.addWidget(label_API_KEY, 0, 0)
        audio_setting_area.addWidget(self.input_API_KEY, 0, 1)
        audio_setting_area.addWidget(label_SECRET_KEY, 0, 2)
        audio_setting_area.addWidget(self.input_SECRET_KEY, 0, 3)
        audio_setting_area.addWidget(label_silence_thresh, 1, 0)
        audio_setting_area.addWidget(self.input_silence_thresh, 1, 1)
        audio_setting_area.addWidget(label_min_silence_len, 1, 2)
        audio_setting_area.addWidget(self.input_min_silence_len, 1, 3)

        audio_setting_box.setFixedHeight(120)
        audio_setting_layout.addWidget(audio_setting_box)
        audio_setting.setLayout(audio_setting_layout)


        # 底部开始按钮布局
        bottom = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignBottom)
        bottom_layout.setAlignment(Qt.AlignRight)
        save_settings_button = QPushButton("保存")
        save_settings_button.clicked.connect(self.save_settings)
        save_settings_button.setFixedSize(80, 35)
        save_settings_button.setStyleSheet("font-size:16px;")
        bottom_layout.addWidget(save_settings_button)
        bottom.setLayout(bottom_layout)

        main_layout.addWidget(video_setting)
        main_layout.addWidget(audio_setting)
        main_layout.addWidget(bottom)

    def save_settings(self):
        settings["name"] = "识别参数"
        settings["video"]["frame_frequency"] = int(self.input_frame_frequency.text())
        settings["video"]["h1"] = float(self.input_h1.text())
        settings["video"]["h2"] = float(self.input_h2.text())
        settings["audio"]["API_KEY"] = self.input_API_KEY.text()
        settings["audio"]["SECRET_KEY"] = self.input_SECRET_KEY.text()
        settings["audio"]["silence_thresh"] = int(self.input_silence_thresh.text())
        settings["audio"]["min_silence_len"] = int(self.input_min_silence_len.text())
        try:
            with open('settings.json', 'w', encoding='utf8') as f:
                json.dump(settings, f)
            QMessageBox.information(self, "提示", "保存成功", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.warning(self, "询问", "保存失败", QMessageBox.Ok)

    def insert_output(self, message):
        output_line = "{} {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
        text_cursor = self.text_output_content.textCursor()
        text_cursor.setPosition(0)
        self.text_output_content.setTextCursor(text_cursor)
        self.text_output_content.insertPlainText(output_line)
        self.text_output_content.update()

    def insert_result(self, message):
        # text_cursor = self.text_result_content.textCursor()
        # text_cursor.setPosition(0)
        # self.text_result_content.setTextCursor(text_cursor)
        self.text_result_content.insertPlainText(message + "\n\n")
        self.text_result_content.update()

    def insert_content(self, words):
        self.contents.append(words)

    def change_status_bar_message(self, msg):
        """修改状态栏回调"""
        self.status_bar.showMessage(msg)

    def change_btn_status(self, btn_status):
        self.start_button.setDisabled(btn_status)

    def open_file(self):
        file_name = QFileDialog.getOpenFileNames(self, '打开文件', '.', '视频文件(*.mp4 *.avi)')
        file_list = file_name[0]
        self.current_video_path = []
        for file in file_list:
            file_path = file
            self.current_video_path.append(Path(file))
            self.insert_text_signal.emit("导入视频文件: {}".format(file_path))
        self.video_path_input.setText(",".join([str(i) for i in self.current_video_path]))


    def save_file(self):
        file_name = QFileDialog.getSaveFileName(self, '保存文件', '.', '文本文件(*.txt)')
        file_path = WindowsPath(file_name[0])
        text = "\n".join(self.contents)
        try:
            with open(file_path, 'w', encoding='utf8') as f:
                f.write(text)
        except PermissionError as e:
            pass
        self.insert_text_signal.emit("导出成功: {}".format(file_path))

    def match_word(self):
        word = self.match_input.text()
        replace_contents = []
        for i in self.contents:
            if word in i:
                line = i.replace(word, '<span style="background-color:yellow">{}</span>'.format(word))
            else:
                line = i
            replace_contents.append(line)
        results = "".join(["<p>{}</p>".format(i) for i in replace_contents])
        self.text_result_content.setHtml(results)

    def refresh(self):
        results = "".join(["<p>{}</p>".format(i) for i in self.contents])
        self.text_result_content.setHtml(results)

    def tab_on_click(self, index):
        pass

    def cb_on_click(self, index):
        if index == 0:
            self.mode = VIDEO
        elif index == 1:
            self.mode = AUDIO

    def execute(self):
        self.contents = []  # 清空当前内容
        self.text_result_content.clear()  # 清空显示识别内容
        if not self.current_video_path:
            QMessageBox.information(self, "询问", "请选择需要识别的视频文件", QMessageBox.Ok)
        else:
            self.start_btn_signal.emit(True)
            for i in self.current_video_path:
                self.task_queue.put(i)
            if self.mode == AUDIO:
                self.speech_recognition_thead = SpeechRecognition(self.insert_text_signal, self.insert_result_signal,
                                                                  self.insert_content_signal, self.start_btn_signal,
                                                                  self.task_queue)
                self.speech_recognition_thead.start()
            else:
                self.image_recognition_thead = ImageRecognition(self.insert_text_signal, self.insert_result_signal,
                                                                self.insert_content_signal, self.start_btn_signal,
                                                                self.task_queue)
                self.image_recognition_thead.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())





































