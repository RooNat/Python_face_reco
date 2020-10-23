import cv2
import numpy as np
from PyQt5 import QtCore,QtGui
from PyQt5.QtCore import QTimer, QRegExp, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QRegExpValidator, QTextCursor
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMessageBox
from PyQt5.uic import loadUi

import logging        #日志
import logging.config
import queue
import threading
import sqlite3
import os
import sys
import time

from datetime import datetime   #用于获取当前时间

# 用户取消了更新数据库操作
class OperationCancel(Exception):
    pass

# 采集过程中出现干扰
class RecordDisturbance(Exception):
    pass

class FaceDataRecordUI(QWidget): #设置DataRecordUI类
    receiveLogSignal = pyqtSignal(str)  # 使用pyqtSinal定义新的str信号
    def __init__(self):       #__init__方法
        super(FaceDataRecordUI, self).__init__()  #构造函数初始化
        loadUi('./ui/face_dataRecord.ui', self) #下载UI界面
        self.setWindowIcon(QIcon('./icons/icon.png'))  #设置图标
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
        # OpenCV
        self.cap = cv2.VideoCapture()  # 设计opencv类 创建一个VideoCapture类的实例，如果传入对应的参数，可以直接打开视频文件或者要调用的摄像头。

        self.logQueue = queue.Queue()  # 日志队列

        # 图像捕获
        self.isExternalCameraUsed = False  # 标志 图像捕获标志
        self.OpenVideoButton.toggled.connect(self.startWebcam)  # 类似于打开和关闭的信号，strtWebcam为函数（打开或关闭摄像头）
        self.OpenVideoButton.setCheckable(True)  # 打开摄像头的按钮
        self.TakephotoButton.clicked.connect(self.photo_face)

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)


        # 数据库
        self.database = './FaceRecobase.db'  # 数据库名称
        self.datasets = './photo'  # 数据保存的文件
        self.isDbReady = False  # 标志

        # 用户信息
        self.isUserInfoReady = False
        self.userInfo = {'stu_id': '', 'cn_name': '', 'en_name': ''}
        self.addUserButton.clicked.connect(self.addOrUpdateUserInfo)  # 增加与修改用户资料的函数
        self.migrationButton.clicked.connect(self.migrateToDb)  # 同步数据库的按钮与同步数据库的函数

        self.isFaceDataReady = False

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()
        self.initDb()

    def close_window(self):
            self.close()
# 打开/关闭摄像头
    def startWebcam(self, status):
        if status:
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            ret, frame = self.cap.read()

            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.OpenVideoButton.setIcon(QIcon('./icons/error.png'))
                self.OpenVideoButton.setChecked(False)
            else:
                self.OpenVideoButton.setText('关闭摄像头')
                self.TakephotoButton.setEnabled(True)
                self.timer.start(5)
                self.OpenVideoButton.setIcon(QIcon('./icons/success.png'))
        else:
            if self.cap.isOpened():
                if self.timer.isActive():
                    self.timer.stop()
                self.cap.release()
                self.Capturelabel.clear()
                self.Capturelabel.setText('<font color=black>摄像头未开启</font>')
                self.OpenVideoButton.setText('打开摄像头')
                self.TakephotoButton.setEnabled(False)
                self.OpenVideoButton.setIcon(QIcon())


    # 定时器，实时更新画面
    def updateFrame(self):
        ret, frame = self.cap.read()      #从摄像头读取照片
        if ret:
            self.displayImage(frame)


    # 显示图像
    def displayImage(self, img):
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

        self.outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.Capturelabel.setPixmap(QPixmap.fromImage(self.outImage))
        self.Capturelabel.setScaledContents(True)

    #拍照
    def photo_face(self):
        photo_save_path = os.path.join(os.path.dirname(os.path.abspath('__file__')),
                                       'photo/')
        self.outImage.save(photo_save_path + self.userInfo.get('cn_name') + ".jpg")

        QMessageBox.information(self, "Information",
                                self.tr("拍照成功!"))
        self.isFaceDataReady = True
        self.migrationButton.setEnabled(True)
    # 初始化数据库
    def initDb(self):
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()
        try:
            # 检测人脸数据目录是否存在，不存在则创建
            if not os.path.isdir(self.datasets):
                os.makedirs(self.datasets)

            # 查询数据表是否存在，不存在则创建
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              stu_id VARCHAR(10) PRIMARY KEY NOT NULL,
                              face_id INTEGER DEFAULT -1,
                              cn_name VARCHAR(10) NOT NULL,
                              en_name VARCHAR(16) NOT NULL,
                              created_time DATE DEFAULT (date('now','localtime'))
                              )
                          ''')
            # 查询数据表记录数
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            #self.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：初始化数据库失败')
        else:
            self.isDbReady = True
            self.UserlcdNumber.display(dbUserCount)
            self.logQueue.put('Success：数据库初始化完成')
            self.addUserButton.setEnabled(True)
        finally:
            cursor.close()
            conn.commit()
            conn.close()

    # 增加/修改用户信息
    def addOrUpdateUserInfo(self):
        self.userInfoDialog = FaceUserInfoDialog()

        stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
            'en_name')
        self.userInfoDialog.UsernumlineEdit.setText(stu_id)
        self.userInfoDialog.UsernamelineEdit.setText(cn_name)
        self.userInfoDialog.EnglishnamelineEdit.setText(en_name)

        self.userInfoDialog.okButton.clicked.connect(self.checkToApplyUserInfo)
        self.userInfoDialog.exec()

    # 校验用户信息并提交
    def checkToApplyUserInfo(self):
        #global enname
        if not (self.userInfoDialog.UsernumlineEdit.hasAcceptableInput() and
                self.userInfoDialog.UsernamelineEdit.hasAcceptableInput() and
                self.userInfoDialog.EnglishnamelineEdit.hasAcceptableInput()):
            self.userInfoDialog.msgLabel.setText('<font color=red>你的输入有误，提交失败，请检查并重试！</font>')
        else:
            # 获取用户输入
            self.userInfo['stu_id'] = self.userInfoDialog.UsernumlineEdit.text().strip()
            self.userInfo['cn_name'] = self.userInfoDialog.UsernamelineEdit.text().strip()
            self.userInfo['en_name'] = self.userInfoDialog.EnglishnamelineEdit.text().strip()

            # 信息确认
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            self.UsernumlineEdit.setText(stu_id)
            self.UsernamelineEdit.setText(cn_name)
            self.EnglishnamelineEdit.setText(en_name)
            #enname=en_name
            self.isUserInfoReady = True
            self.migrationButton.setIcon(QIcon())

            # 关闭对话框
            self.userInfoDialog.close()

    # 同步用户信息到数据库
    def migrateToDb(self):
        if self.isFaceDataReady:
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                if cursor.fetchall():
                    text = '数据库已存在学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                    informativeText = '<b>是否覆盖？</b>'
                    ret = FaceDataRecordUI.callDialog(QMessageBox.Warning, text, informativeText,
                                                  QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        # 更新已有记录
                        cursor.execute('UPDATE users SET cn_name=?, en_name=? WHERE stu_id=?',
                                       (cn_name, en_name, stu_id,))
                    else:
                        raise OperationCancel  # 记录取消覆盖操作
                else:
                    # 插入新记录
                    cursor.execute('INSERT INTO users (stu_id, cn_name, en_name) VALUES (?, ?, ?)',
                                   (stu_id, cn_name, en_name,))

                cursor.execute('SELECT Count(*) FROM users')
                result = cursor.fetchone()
                dbUserCount = result[0]
            except OperationCancel:
                pass
            except Exception as e:
                logging.error('读写数据库异常，无法向数据库插入/更新记录')
                self.migrationButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，同步失败')
            else:
                text = '<font color=blue>{}</font> 已添加/更新到数据库。'.format(stu_id)
                informativeText = '<b><font color=blue>{}</font> 的人脸数据采集已完成！</b>'.format(cn_name)
                FaceDataRecordUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                # 清空用户信息缓存
                for key in self.userInfo.keys():
                    self.userInfo[key] = ''
                self.isUserInfoReady = False
                self.isFaceDataReady = False

                # 清空历史输入
                self.UsernumlineEdit.clear()
                self.UsernamelineEdit.clear()
                self.EnglishnamelineEdit.clear()
                self.migrationButton.setIcon(QIcon('./icons/success.png'))

                # 允许继续增加新用户
                self.addUserButton.setEnabled(True)
                self.migrationButton.setEnabled(False)

            finally:
                cursor.close()
                conn.commit()
                conn.close()
        else:
            self.logQueue.put('Error：操作失败，你尚未完成人脸数据采集')
            self.migrationButton.setIcon(QIcon('./icons/error.png'))

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
        msg.setWindowTitle('OpenCV Face Recognition System - DataRecord')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()
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
    # 窗口关闭事件，关闭定时器、摄像头
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# 用户信息填写对话框
class FaceUserInfoDialog(QDialog):
    def __init__(self):
        super(FaceUserInfoDialog, self).__init__()
        loadUi('./ui/UserInfoDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;}QPushButton:hover{background:green;}''')
        self.left_close.clicked.connect(self.close_window)  # 关闭窗口
        self.left_mini.clicked.connect(self.showMinimized)  # 最小化窗口
        # 使用正则表达式限制用户输入
        stu_id_regx = QRegExp('^[0-9]{10}$')
        stu_id_validator = QRegExpValidator(stu_id_regx, self.UsernumlineEdit)
        self.UsernumlineEdit.setValidator(stu_id_validator)

        cn_name_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        cn_name_validator = QRegExpValidator(cn_name_regx, self.UsernamelineEdit)
        self.UsernamelineEdit.setValidator(cn_name_validator)

        en_name_regx = QRegExp('^[ A-Za-z]{1,16}$')
        en_name_validator = QRegExpValidator(en_name_regx, self.EnglishnamelineEdit)
        self.EnglishnamelineEdit.setValidator(en_name_validator)

    def close_window(self):
        self.close()

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

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = FaceDataRecordUI()
    window.show()
    sys.exit(app.exec())
