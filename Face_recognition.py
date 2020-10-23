import telegram
import cv2
import dlib
import pygame
import numpy as np
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
# 找不到已训练的人脸数据文件
class TrainingDataNotFoundError(FileNotFoundError):
    pass


# 找不到数据库文件
class DatabaseNotFoundError(FileNotFoundError):
    pass


class CoreUI(QMainWindow):
    database = './Facebase.db'
    trainingData = './recognizer/trainingData.yml'
    cap = cv2.VideoCapture()
    captureQueue = queue.Queue()  # 图像队列
    alarmQueue = queue.LifoQueue()  # 报警队列，后进先出
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG信号

    def __init__(self):
        super(CoreUI, self).__init__()
        loadUi('./ui/Core.ui', self)
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
        # 图像捕获
        self.isExternalCameraUsed = False
        #self.useExternalCameraCheckBox.stateChanged.connect(
           # lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.faceProcessingThread = FaceProcessingThread()
        self.Openvideo.clicked.connect(self.startWebcam)
        self.Voicerecognition.clicked.connect(self.Voicereco)
        self.Voicevideo.clicked.connect(self.video_announce)
        # 数据库
        #self.initdb.setIcon(QIcon('./icons/warning.png'))
        #self.initdb.clicked.connect(self.initDb)
        self.initDb()
        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.updateFrame)

        # 功能开关
        self.faceTrackerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceTracker(self))
        self.faceRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceRecognizer(self))
        self.panalarmCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enablePanalarm(self))

        # 直方图均衡化
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableEqualizeHist(self))

        # 调试模式
        self.debugCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enableDebug(self))
        self.confidenceThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setConfidenceThreshold(self))
        self.autoAlarmThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setAutoAlarmThreshold(self))

        # 报警系统
        self.alarmSignalThreshold = 10
        self.panalarmThread = threading.Thread(target=self.recieveAlarm, daemon=True)
        self.isBellEnabled = True
        self.bellCheckBox.stateChanged.connect(lambda: self.enableBell(self.bellCheckBox))
        self.isTelegramBotPushEnabled = False

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
            if not os.path.isfile(self.trainingData):
                raise TrainingDataNotFoundError

            conn = sqlite3.connect(self.database) #连接数据库
            cursor = conn.cursor() #使用cursor()方法获取操作游标
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DatabaseNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            #self.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except TrainingDataNotFoundError:
            logging.error('系统找不到已训练的人脸数据{}'.format(self.trainingData))
            #self.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现已训练的人脸数据文件，请完成训练后继续')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            #elf.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
                #self.initdb.setIcon(QIcon('./icons/warning.png'))
            else:
                self.logQueue.put('Success：数据库状态正常，发现用户数：{}'.format(dbUserCount))
                #self.initdb.setIcon(QIcon('./icons/success.png'))
                #self.initdb.setEnabled(False)
                self.faceRecognizerCheckBox.setToolTip('须先开启人脸跟踪')
                self.faceRecognizerCheckBox.setEnabled(True)
    # 打开/关闭摄像头
    def startWebcam(self):
        if not self.cap.isOpened():
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()
            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                #self.OpenVideoButton.setIcon(QIcon('./icons/error.png'))
            else:
                self.faceProcessingThread.start()  # 启动OpenCV图像处理线程
                self.timer.start(5)  # 启动定时器
                self.panalarmThread.start()  # 启动报警系统线程
                #self.OpenVideoButton.setIcon(QIcon('./icons/success.png'))
                #self.OpenVideoButton.setText('关闭摄像头')

        else:
            text = '如果关闭摄像头，须重启程序才能再次打开。'
            informativeText = '<b>是否继续？</b>'
            ret = CoreUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

            if ret == QMessageBox.Yes:
                self.faceProcessingThread.stop()
                if self.cap.isOpened():
                    if self.timer.isActive():
                        self.timer.stop()
                    self.cap.release()

                self.Capturelabel.clear()
                self.Capturelabel.setText('<font color=red>摄像头未开启</font>')
                #self.OpenVideoButton.setText('摄像头已关闭')
                #self.OpenVideoButton.setEnabled(False)
                #self.OpenVideoButton.setIcon(QIcon())

    # 定时器，实时更新画面
    def updateFrame(self):
        if self.cap.isOpened():
            # ret, frame = self.cap.read()
            # if ret:
            #     self.showImg(frame, self.Capturelabel)
            if not self.captureQueue.empty():
                captureData = self.captureQueue.get()
                realTimeFrame = captureData.get('realTimeFrame')
                self.displayImage(realTimeFrame, self.Capturelabel)

    # 显示图片
    def displayImage(self, img, qlabel):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))
        qlabel.setScaledContents(True)  # 图片自适应大小

    # 报警系统：是否允许设备响铃
    def enableBell(self, bellCheckBox):
        if bellCheckBox.isChecked():
            self.isBellEnabled = True
            self.statusBar().showMessage('设备发声：开启')
        else:
            if self.isTelegramBotPushEnabled:
                self.isBellEnabled = False
                self.statusBar().showMessage('设备发声：关闭')
            else:
                self.logQueue.put('Error：操作失败，至少选择一种报警方式')
                self.bellCheckBox.setCheckState(Qt.Unchecked)
                self.bellCheckBox.setChecked(True)
        # print('isBellEnabled：', self.isBellEnabled)

    # 设备响铃进程
    @staticmethod
    def bellProcess(queue):
        logQueue = queue
        logQueue.put('Info：设备正在响铃...')
        winsound.PlaySound('./alarm.wav', winsound.SND_FILENAME)

    # 报警系统服务常驻，接收并处理报警信号
    def recieveAlarm(self):
        while True:
            jobs = []
            # print(self.alarmQueue.qsize())
            if self.alarmQueue.qsize() > self.alarmSignalThreshold:  # 若报警信号触发超出既定计数，进行报警
                if not os.path.isdir('./unknown'):
                    os.makedirs('./unknown')
                lastAlarmSignal = self.alarmQueue.get()
                timestamp = lastAlarmSignal.get('timestamp')
                img = lastAlarmSignal.get('img')
                # 疑似陌生人脸，截屏存档
                cv2.imwrite('./unknown/{}.jpg'.format(timestamp), img)
                logging.info('报警信号触发超出预设计数，自动报警系统已被激活')
                self.logQueue.put('Info：报警信号触发超出预设计数，自动报警系统已被激活')

                # 是否进行响铃
                if self.isBellEnabled:
                    p1 = multiprocessing.Process(target=CoreUI.bellProcess, args=(self.logQueue,))
                    p1.start()
                    jobs.append(p1)
                # 等待本轮报警结束
                for p in jobs:
                    p.join()

                # 重置报警信号
                with self.alarmQueue.mutex:
                    self.alarmQueue.queue.clear()
            else:
                continue

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
    def video_announce(self):
        global stuname
        if stuname=='unknown':
            self.baidu_voice("无法识别人脸")
        #if self.stuname!=' '&self.stuname!='unknown':
        else:
            print(stuname)
            voicestr='验证通过 欢迎'+stuname+'的到来'
            print(voicestr)
            self.baidu_voice(voicestr)



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
    def openbrowser(self,text):
        if text =='开始检测。':
            self.startWebcam()
        elif text=='关闭摄像头。':
            self.startWebcam()
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

    def Voicereco(self):
        self.my_record()
        TOKEN = self.getToken(HOST)
        speech = self.get_audio(FILEPATH)
        result = self.speech2text(speech, TOKEN, 1537)
        print(result)
        if type(result) == str:
            self.openbrowser(result.strip('，'))
    # 窗口关闭事件，关闭OpenCV线程、定时器、摄像头
    def closeEvent(self, event):
        if self.faceProcessingThread.isRunning:
            self.faceProcessingThread.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# OpenCV线程
class FaceProcessingThread(QThread):
    def __init__(self):
        super(FaceProcessingThread, self).__init__()
        self.isRunning = True

        self.isFaceTrackerEnabled = True
        self.isFaceRecognizerEnabled = False
        self.isPanalarmEnabled = True

        self.isDebugMode = False
        self.confidenceThreshold = 50
        self.autoAlarmThreshold = 65

        self.isEqualizeHistEnabled = False

    # 是否开启人脸跟踪
    def enableFaceTracker(self, coreUI):
        if coreUI.faceTrackerCheckBox.isChecked():
            self.isFaceTrackerEnabled = True
        else:
            self.isFaceTrackerEnabled = False


    # 是否开启人脸识别
    def enableFaceRecognizer(self, coreUI):
        if coreUI.faceRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isFaceRecognizerEnabled = True
            else:
                CoreUI.logQueue.put('Error：操作失败，请先开启人脸跟踪')
                coreUI.faceRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.faceRecognizerCheckBox.setChecked(False)
        else:
            self.isFaceRecognizerEnabled = False

    # 是否开启报警系统
    def enablePanalarm(self, coreUI):
        if coreUI.panalarmCheckBox.isChecked():
            self.isPanalarmEnabled = True
           # coreUI.statusBar().showMessage('报警系统：开启')
        else:
            self.isPanalarmEnabled = False
            #coreUI.statusBar().showMessage('报警系统：关闭')

    # 是否开启调试模式
    def enableDebug(self, coreUI):
        if coreUI.debugCheckBox.isChecked():
            self.isDebugMode = True
           # coreUI.statusBar().showMessage('调试模式：开启')
        else:
            self.isDebugMode = False
           # coreUI.statusBar().showMessage('调试模式：关闭')

    # 设置置信度阈值
    def setConfidenceThreshold(self, coreUI):
        if self.isDebugMode:
            self.confidenceThreshold = coreUI.confidenceThresholdSlider.value()
            confidence=str(self.confidenceThreshold)
            self.confidencelabel.setText(confidence)
            #coreUI.statusBar().showMessage('置信度阈值：{}'.format(self.confidenceThreshold))

    # 设置自动报警阈值
    def setAutoAlarmThreshold(self, coreUI):
        if self.isDebugMode:
            self.autoAlarmThreshold = coreUI.autoAlarmThresholdSlider.value()
            autoalarm=str(self.autoAlarmThreshold)
            self.alarmlabel.setText(autoalarm)
           # coreUI.statusBar().showMessage('自动报警阈值：{}'.format(self.autoAlarmThreshold))

    # 直方图均衡化
    def enableEqualizeHist(self, coreUI):
        if coreUI.equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
            #coreUI.statusBar().showMessage('直方图均衡化：开启')
        else:
            self.isEqualizeHistEnabled = False
            #coreUI.statusBar().showMessage('直方图均衡化：关闭')

    def run(self):
        #faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        # 帧数、人脸ID初始化
        frameCounter = 0
        currentFaceID = 0
        global stuname
        # 人脸跟踪器字典初始化
        faceTrackers = {}

        isTrainingDataLoaded = False
        isDbConnected = False

        while self.isRunning:
            if CoreUI.cap.isOpened():
                ret, frame = CoreUI.cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 是否执行直方图均衡化
                if self.isEqualizeHistEnabled:
                    gray = cv2.equalizeHist(gray)
                #faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90)) #人脸检测
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                self.faceCascade.setInput(blob)
                faces = self.faceCascade.forward()
                h1, w1, c1 = frame.shape
                # 预加载数据文件
                if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):
                    recognizer = cv2.face.LBPHFaceRecognizer_create()     #LBPH识别器
                    #recognizer = cv2.face.FisherFaceRecognizer_create()
                    recognizer.read(CoreUI.trainingData)
                    isTrainingDataLoaded = True
                if not isDbConnected and os.path.isfile(CoreUI.database):
                    conn = sqlite3.connect(CoreUI.database)
                    cursor = conn.cursor()
                    isDbConnected = True

                captureData = {}
                realTimeFrame = frame.copy()
                alarmSignal = {}
                # 人脸跟踪
                if self.isFaceTrackerEnabled:

                    # 要删除的人脸跟踪器列表初始化
                    fidsToDelete = []

                    for fid in faceTrackers.keys():
                        # 实时跟踪
                        trackingQuality = faceTrackers[fid].update(realTimeFrame)
                        # 如果跟踪质量过低，删除该人脸跟踪器
                        if trackingQuality < 7:
                            fidsToDelete.append(fid)
                    # 删除跟踪质量过低的人脸跟踪器
                    for fid in fidsToDelete:
                        faceTrackers.pop(fid, None)
                    # for (_x, _y, _w, _h) in faces:
                    for i in range(0, faces.shape[2]):
                        confidence1 = faces[0, 0, i, 2]
                        isKnown = False
                        if confidence1 > 0.6:
                            box = faces[0, 0, i, 3:7] * np.array([w1, h1, w1, h1])
                            (startX, startY, endX, endY) = box.astype("int")
                            if self.isFaceRecognizerEnabled:
                                cv2.rectangle(realTimeFrame, (startX, startY), (endX, endY), (255, 0, 255), 2)
                                face_id, confidence = recognizer.predict(gray[startY:endY, startX:endX])
                                logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                                if self.isDebugMode:
                                    CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                            # 从数据库中获取识别人脸的身份信息
                                try:
                                    cursor.execute("SELECT * FROM users WHERE face_id=?", (face_id,))
                                    result = cursor.fetchall()
                                    if result:
                                        en_name = result[0][3]
                                        stuname=result[0][2]
                                    else:
                                        raise Exception
                                except Exception as e:
                                    logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                    CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                    en_name = ''

                            # 若置信度评分小于置信度阈值，认为是可靠识别
                                if confidence < self.confidenceThreshold:
                                    isKnown = True
                                    cv2.putText(realTimeFrame, en_name, (startX - 5, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 97, 255), 2)
                                    print(stuname)
                                    #ft = ft2.put_chinese_text('msyh.ttf')
                                    #ft.draw_text(realTimeFrame, (_x-5, _y-10), en_name, 25, (0, 0, 255))
                                else:
                                    # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                                    cv2.putText(realTimeFrame, 'unknown', (startX - 5, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 2)
                                    stuname='unknown'
                                    # 若置信度评分超出自动报警阈值，触发报警信号
                                    if confidence > self.autoAlarmThreshold:
                                    # 检测报警系统是否开启
                                         if self.isPanalarmEnabled:
                                              alarmSignal['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')
                                              alarmSignal['img'] = realTimeFrame
                                              CoreUI.alarmQueue.put(alarmSignal)
                                              logging.info('系统发出了报警信号')

                            # 帧数自增
                            frameCounter += 1

                            # 每读取10帧，检测跟踪器的人脸是否还在当前画面内
                            if frameCounter % 10 == 0:
                               # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                               # 而dlib人脸跟踪器要求的是int类型
                               x = int(startX)
                               y = int(startY)
                               w = int(endX-startX)
                               h = int(endY-startY)

                               # 计算中心点
                               x_bar = x + 0.5 * w
                               y_bar = y + 0.5 * h

                               # matchedFid表征当前检测到的人脸是否已被跟踪
                               matchedFid = None

                               for fid in faceTrackers.keys():
                                  # 获取人脸跟踪器的位置
                                  # tracked_position 是 dlib.drectangle 类型，用来表征图像的矩形区域，坐标是浮点数
                                  tracked_position = faceTrackers[fid].get_position()
                                  # 浮点数取整
                                  t_x = int(tracked_position.left())
                                  t_y = int(tracked_position.top())
                                  t_w = int(tracked_position.width())
                                  t_h = int(tracked_position.height())

                                  # 计算人脸跟踪器的中心点
                                  t_x_bar = t_x + 0.5 * t_w
                                  t_y_bar = t_y + 0.5 * t_h

                                  # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                                  # 说明当前人脸已被跟踪
                                  if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                          (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                      matchedFid = fid

                              # 如果当前检测到的人脸是陌生人脸且未被跟踪
                               if not isKnown and matchedFid is None:
                                  # 创建一个人脸跟踪器
                                  tracker = dlib.correlation_tracker()
                                  # 锁定跟踪范围
                                  tracker.start_track(realTimeFrame, dlib.rectangle(x , y , x + w , y + h ))
                                  # 将该人脸跟踪器分配给当前检测到的人脸
                                  faceTrackers[currentFaceID] = tracker
                                  # 人脸ID自增
                                  currentFaceID += 1

                    # 使用当前的人脸跟踪器，更新画面，输出跟踪结果
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # 在跟踪帧中圈出人脸
                        cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                        cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                                    2)

                captureData['originFrame'] = frame
                captureData['realTimeFrame'] = realTimeFrame
                CoreUI.captureQueue.put(captureData)

            else:
                continue

    # 停止OpenCV线程
    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = CoreUI()
    window.show()
    sys.exit(app.exec())
