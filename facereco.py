import telegram
import cv2
import face_recognition
import dlib
import pygame
import numpy as np
from os import listdir,getcwd
import requests
import json
import base64
import os
import logging
import speech_recognition as sr
from playsound import playsound
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtCore,QtGui
from PIL import Image, ImageDraw, ImageFont #与cv2 进行转换PIl可以显示汉字cv2不行
import os
import webbrowser
import logging
import logging.config
import sqlite3
import sys
import threading
import queue
import multiprocessing
import winsound
import wave
import requests
import time
import base64
from pyaudio import PyAudio, paInt16
import webbrowser
from aip import AipSpeech
framerate = 16000  # 采样率
num_samples = 2000  # 采样点
channels = 1  # 声道
sampwidth = 2  # 采样宽度2bytes
FILEPATH = 'speech.wav'
stuname=' '
base_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s"
APIKey = "MoxFLhOWqhfmEGSvxHQ2UI5n"
SecretKey = "UtLtVbVRvQnmRQcaeAbYOoovifuTeMQz"
APPID="22735458"
HOST = base_url % (APIKey, SecretKey)
from configparser import ConfigParser
from datetime import datetime

client = AipSpeech(APPID, APIKey, SecretKey)
# 找不到数据库文件
class DatabaseNotFoundError(FileNotFoundError):
    pass


class FaceCoreUI(QMainWindow):
    cap = cv2.VideoCapture()
    captureQueue = queue.Queue()  # 图像队列
    #alarmQueue = queue.LifoQueue()  # 报警队列，后进先出
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG信号
    need_record_name1=([])
    def __init__(self):
        super(FaceCoreUI, self).__init__()
        loadUi('./ui/face_recognition_record.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        #self.setFixedSize(1161, 623)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;}QPushButton:hover{background:green;}''')
        self.left_close.clicked.connect(self.close_window)  # 关闭窗口
        self.left_mini.clicked.connect(self.showMinimized)  # 最小化窗口
        self.confidenceThreshold=60
        self.setsizeThreshold=30
        # 图像捕获
        #self.isExternalCameraUsed = False
        self.video_btn = 0
        #self.faceProcessingThread = FaceProcessingThread()
        #self.isFaceRecognizerEnabled=False
        self.Openvideo.clicked.connect(self.btn_open_cam_click)
        self.Voicerecognition.clicked.connect(self.Voicereco)
        self.Voicevideo.clicked.connect(self.video_announce)
        self.Facerecognition.clicked.connect(self.face_recognition_btn)
        self.confidenceThresholdSlider.valueChanged.connect(self.setconfidence)
        self.setsizeSlider.valueChanged.connect(self.setsize)
        # 数据库
        self.timer = QTimer(self)  # 初始化一个定时器
        #self.timer.timeout.connect(self.updateFrame)
        self.database = './FaceRecobase.db'
        self.initDb()
        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()


    def close_window(self):
            self.close()
    # 检查数据库状态
    def initDb(self):
        try:
            if not os.path.isfile(self.database):
                raise DatabaseNotFoundError

            conn = sqlite3.connect(self.database) #连接数据库
            cursor = conn.cursor() #使用cursor()方法获取操作游标
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DatabaseNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
            else:
                self.logQueue.put('Success：数据库状态正常，发现用户数：{}'.format(dbUserCount))


    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    # LOG输出
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logtextEdit.moveCursor(QTextCursor.End)
        self.logtextEdit.insertPlainText(log)
        self.logtextEdit.ensureCursorVisible()  # 自动滚屏

    # 系统对话框
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - Core')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    def getToken(self,host):
        res = requests.post(host)
        return res.json()['access_token']

    #保存文件
    def save_wave_file(self,filepath, data):
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b''.join(data))
        wf.close()


    def baidu_voice(self,voice):
        """ 你的 APPID AK SK """
        try:
            result = client.synthesis(voice, 'zh', 1, {
                'vol': 8, 'per': 0
            })
            # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
            if not isinstance(result, dict):
                with open('auido.mp3', 'wb') as f:
                    f.write(result)
                #os.system('auido.mp3')
            #self.playMusic('auido.mp3')
            playsound('auido.mp3')
            os.remove('auido.mp3')
            #os.system('auido.mp3')
        except:
            return "语音合成出现问题 请检查"
    #语音播报
    def video_announce(self):
        if self.set_name.issubset(self.need_record_name1):  # 如果self.set_names是self.need_record_names 的子集返回ture
            pass  # need_record_name1是要写进excel中的名字信息 set_name是从摄像头中读出人脸的tuple形式
        else:
            self.different_name1 = self.set_name.difference(
                self.need_record_name1)  # 获取到self.set_name有 而self.need_record_name 无的名字
            self.need_record_name1 = self.set_name.union(self.need_record_name1)  # 把self.need_record_name  变成两个集合的并集
            # different_name是为了获取到之前没有捕捉到的人脸  并且再次将need_recore_name1进行更新
        try:
            need_voice_name = list(self.need_record_name1)
        except:
            need_voice_name = []
        if need_voice_name != []:
            print(need_voice_name)
            if 'Unknown' in need_voice_name:  # 把unknown去掉 不进行播报
                need_voice_name.remove('Unknown')
            tuple_voice_name = tuple(need_voice_name)
            if tuple_voice_name == ():
                self.baidu_voice('还未识别出人脸')
            else:
                voice_str = '验证通过 欢迎'
                for i in tuple_voice_name:
                    voice_str = voice_str + i + ' '
                voice_str = voice_str + '的到来'
                print(voice_str)
                self.baidu_voice(voice_str)  # 欢迎 某 某某 的到来


    #录音
    def my_record(self):
        pa = PyAudio()
        stream = pa.open(format=paInt16, channels=channels,
                         rate=framerate, input=True, frames_per_buffer=num_samples)
        my_buf = []
        # count = 0
        t = time.time()
        print('正在录音...')

        while time.time() < t + 4:  # 秒
            string_audio_data = stream.read(num_samples)
            my_buf.append(string_audio_data)
        print('录音结束.')
        self.save_wave_file(FILEPATH, my_buf)
        stream.close()

    def get_audio(self,file):
        with open(file, 'rb') as f:
            data = f.read()
        return data

    #语音识别音频预处理
    def speech2text(self,speech_data, token, dev_pid=1537):
        FORMAT = 'wav'
        RATE = '16000'
        CHANNEL = 1
        CUID = '*******'
        SPEECH = base64.b64encode(speech_data).decode('utf-8')

        data = {
            'format': FORMAT,
            'rate': RATE,
            'channel': CHANNEL,
            'cuid': CUID,
            'len': len(speech_data),
            'speech': SPEECH,
            'token': token,
            'dev_pid': dev_pid
        }
        url = 'https://vop.baidu.com/server_api'
        headers = {'Content-Type': 'application/json'}
        # r=requests.post(url,data=json.dumps(data),headers=headers)
        print('正在识别...')
        r = requests.post(url, json=data, headers=headers)
        Result = r.json()
        if 'result' in Result:
            return Result['result'][0]
        else:
            return Result

    #语音识别比较
    def openbrowser(self,text):
        if text =='打开摄像。':
            self.btn_open_cam_click()
        elif text=='开始检测。':
            self.face_recognition_btn()
        elif text=='关闭摄像头。':
            self.btn_open_cam_click()
        else:
            self.baidu_voice('不好意思，没听清')

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    #语音识别
    def Voicereco(self):
        self.my_record()
        TOKEN = self.getToken(HOST)
        speech = self.get_audio(FILEPATH)
        result = self.speech2text(speech, TOKEN, 1537)
        print(result)
        if type(result) == str:
            self.openbrowser(result.strip('，'))
    #置信度阈值滚动条
    def setconfidence(self):
        self.confidenceThreshold = self.confidenceThresholdSlider.value()
        confidence=str(self.confidenceThreshold)
        self.confidencelabel.setText(confidence)

    #人脸识别图像处理大小滚动条
    def setsize(self):
        self.setsizeThreshold=self.setsizeSlider.value()
        size=str(self.setsizeThreshold)
        self.setsizelabel.setText(size)

    def btn_open_cam_click(self):  #打开摄像头 按钮函数
        self.source=0
        flag2=self.cap.isOpened()  #判断摄像头是否被打开 如果被打开flag2就是ture反之就是false
        if flag2 == False:
            self.cap.open(self.source)
            try:
                self.show_camera()
            except:
                QMessageBox.about(self, 'warning', '摄像头不能正常被打开')
        else:
            text = '如果关闭摄像头，须重启程序才能再次打开。'
            informativeText = '<b>是否继续？</b>'
            ret = FaceCoreUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.No)
            if ret == QMessageBox.Yes:
                if self.cap.isOpened():
                        print('关闭摄像头')
                        self.cap.release()   # 关闭摄像头 对cap进行释放
                        self.Openvideo.setText(u'打开摄像头')
                        self.Capturelabel.clear()
                        self.Capturelabel.setText('<font color=black>摄像头未开启</font>')


    def face_recognition_btn(self):  # 人脸识别按钮  通过video_btn的值来控制
        flag2=self.cap.isOpened()
        if flag2== False:
            QMessageBox.information(self, "Warning",
                                self.tr("请先打开摄像头!"))
        else:
            #self.time_step=0
            if self.video_btn==0:
                self.video_btn = 1
                self.Facerecognition.setText(u'关闭人脸识别')
                self.show_camera()

                #由于
            elif self.video_btn==1:
                self.video_btn=0
                #self.time_step=0  #进度条初始化 当下次打开人脸识别的时候  再次打开进度条
                self.Facerecognition.setText(u'人脸识别')
                self.qingping()
                self.show_camera()
                self.qingping()

    def qingping(self):  # 不需要显示信息的时候   把显示到信息的那部分清除掉 在循环中保存了几次那些lable就不在发生变化了
        self.person1.clear()
        self.person1name.setText("")  # 信息1
        self.person1stuid.setText("")
        self.person1english.setText("")
        self.person2.clear()
        self.person2name.setText("")
        self.person2stuid.setText("")
        self.person2english.setText("")
        self.person3.clear()
        self.person3name.setText("")
        self.person3stuid.setText("")
        self.person3english.setText("")

    def show_camera(self):  # 展示摄像头画面并进行人脸识别的功能
        if self.video_btn == 0:  # 在前面就设置了video_btn为0 为了在人脸识别的时候直接把这个值给改了 这样人脸识别和摄像头展示就分开了

            self.Openvideo.setText(u'关闭摄像头')

            while (self.cap.isOpened()):
                ret, self.image = self.cap.read()
                QApplication.processEvents()  # 这句代码告诉QT处理来处理任何没有被处理的事件，并且将控制权返回给调用者  让代码变的没有那么卡
                show = cv2.resize(self.image, (720, 640))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.Capturelabel.setPixmap(QPixmap.fromImage(self.showImage))

            #  因为他最后会存留一张 图像在lable上需要对 lable_5进行清理
            self.Capturelabel.clear()
            self.Capturelabel.setText('<font color=black>摄像头未开启</font>')
        elif self.video_btn == 1:
            # 这段代码是 获取photo文件夹中 人的信息
            filepath = 'photo'
            filename_list = listdir(filepath)
            known_face_names = []  # 存储人名
            known_face_encodings = []  # 存储编码
            a = 0
            print('2')
            for filename in filename_list:  # 依次读入列表中的内容
                a += 1
                QApplication.processEvents()  # 可去掉 进度条
                if filename.endswith('jpg'):  # 后缀名'jpg'匹对
                    known_face_names.append(filename[:-4])  # 把文件名字的后四位.jpg去掉获取人名
                    file_str = 'photo' + '/' + filename
                    a_images = face_recognition.load_image_file(file_str)  # 从中获取图片
                    print(file_str)
                    a_face_encoding = face_recognition.face_encodings(a_images)[0]  # 对图片解码
                    known_face_encodings.append(a_face_encoding)
            print(known_face_names, a)
            # knowe_face_names里面放着每个人的名字   known_face_encodings里面放着提取出来的每个人的人脸特征信息

            face_locations = []
            face_encodings = []
            face_names = []
            process_this_frame = True
            while (self.cap.isOpened()):
                SET_SIZE = self.setsizeThreshold * 0.01
                TOLERANCE = self.confidenceThreshold * 0.01
                ret, frame = self.cap.read()
                QApplication.processEvents()
                # 改变摄像头图像的大小，图像小，所做的计算就少
                small_frame = cv2.resize(frame, (0, 0), fx=SET_SIZE, fy=SET_SIZE)

                # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
                rgb_small_frame = small_frame[:, :, ::-1]
                # print('4 is running')
                if process_this_frame:
                    QApplication.processEvents()
                    # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
                    face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0,
                                                                     model="cnn")
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    # print('5 is  running')
                    for face_encoding in face_encodings:
                        # 默认为unknown
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,
                                                                 tolerance=TOLERANCE)
                        # 阈值太低容易造成无法成功识别人脸，太高容易造成人脸识别混淆 默认阈值tolerance为0.6
                        # print(matches)
                        name = "Unknown"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]

                        face_names.append(name)
                process_this_frame = not process_this_frame
                # 将捕捉到的人脸显示出来
                self.set_name = set(face_names)
                self.set_names = tuple(self.set_name)  # 把名字先设为了一个 集合 把重复的去掉 再设为tuple 以便于下面显示其他信息和记录 调用
                #voice_syn = str()
                print(self.set_names)  # 把人脸识别检测到的人 用set_names 这个集合收集起来
                #self.write_record()  # 把名字记录到excel中去
                # self.video_announce()
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # 由于我们检测到的帧被缩放到1/4大小，所以要缩小面位置
                    top *= int(1 / SET_SIZE)
                    right *= int(1 / SET_SIZE)
                    bottom *= int(1 / SET_SIZE)
                    left *= int(1 / SET_SIZE)
                    # 矩形框
                    cv2.rectangle(frame, (left, top), (right, bottom), (60, 20, 220), 3)

                    print('face recognition is running')
                    # def draw_text(self, image, pos, text, text_size, text_color)
                    # 由于 opencv无法显示汉字 之前使用的方法当照片很小时会报错，此次采用了另一种方法使用PIL进行转换
                    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                    pilimg = Image.fromarray(cv2img)
                    draw = ImageDraw.Draw(pilimg)  # 图片上打印
                    font = ImageFont.truetype("msyh.ttf", 27, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                    draw.text((left + 10, bottom), name, (220, 20, 60), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

                    # PIL图片转cv2 图片
                    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                self.show_picture()  # 调用显示详细信息的函数

                show_video = cv2.resize(frame, (720, 640))
                show_video = cv2.cvtColor(show_video, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                self.showImage = QImage(show_video.data, show_video.shape[1], show_video.shape[0], QImage.Format_RGB888)
                self.Capturelabel.setPixmap(QPixmap.fromImage(self.showImage))


    def show_picture(self):  #  在人脸识别的右边显示 识别出来人的详细信息
         if self.video_btn==1:
             conn = sqlite3.connect(self.database)
             cursor = conn.cursor()
             photo_message={0:[self.person1name,self.person1stuid,self.person1english,self.person1],1:[self.person2name,self.person2stuid,self.person2english,self.person2],
                            2:[self.person3name,self.person3stuid,self.person3english,self.person3]}
             #使用photo_message  记录出现人物数以及需要放置的lable 使用下面的遍历 来达到效果
             if len(self.set_names)>3:
                 show_person=3
             else:
                 show_person=len(self.set_names)
             if show_person!=0:
                 for person in range(show_person):
                     try:
                         cn_name=self.set_names[person]   #识别出人物的名字
                         person_name=photo_message[person][0] #信息所对应的lable
                         person_stuid=photo_message[person][1]
                         person_english=photo_message[person][2]
                         person_photo=photo_message[person][3] #照片所对对应的lable
                         if cn_name=='Unknown':
                             en_name='Unknown'
                             stu_id='Unknown'
                         else:
                             cursor.execute("SELECT * FROM users WHERE cn_name=?", (cn_name,))
                             result = cursor.fetchall()
                             stu_id = result[0][0]
                             en_name = result[0][3]
                         name_str = 'photo//' + cn_name + '.jpg'
                         picture = QPixmap(name_str)
                         person_name.setText(cn_name)
                         person_stuid.setText(stu_id)
                         person_english.setText(en_name)
                         person_photo.setPixmap(picture)  # 把照片放到label_7上面去
                         person_photo.setScaledContents(True)  # 让照片能够在label上面自适应大小
                     except :
                         QMessageBox.about(self,'warning','请检查'+cn_name+'的信息')

    # 窗口关闭事件，关闭OpenCV线程、定时器、摄像头
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = FaceCoreUI()
    window.show()
    sys.exit(app.exec())
