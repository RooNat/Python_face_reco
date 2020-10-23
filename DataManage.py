import cv2
import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QTableWidgetItem, QAbstractItemView
from PyQt5.uic import loadUi
from PyQt5 import QtCore,QtGui
import logging
import logging.config
import os
import shutil
import sqlite3
import sys
import threading
import multiprocessing

from datetime import datetime


# 自定义数据库记录不存在异常
class RecordNotFound(Exception):
    pass


class DataManageUI(QWidget):
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # 日志信号

    def __init__(self):
        super(DataManageUI, self).__init__()
        loadUi('./ui/dataManage.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        #self.setFixedSize(931, 577)

        # 设置tableWidget只读，不允许修改
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
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
        # 数据库
        self.database = './Facebase.db'
        self.datasets = './datasets'
        self.isDbReady = False
        self.initdb.clicked.connect(self.initDb)

        # 用户管理
        self.QueryButton.clicked.connect(self.queryUser)
        self.deleteUserButton.clicked.connect(self.deleteUser)

        # 直方图均衡化
        self.isEqualizeHistEnabled = False
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.enableEqualizeHist(self.equalizeHistCheckBox))

        # 训练人脸数据
        self.trainButton.clicked.connect(self.train)

        # 系统日志
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()
    def close_window(self):
            self.close()
    # 是否执行直方图均衡化
    def enableEqualizeHist(self, equalizeHistCheckBox):
        if equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
        else:
            self.isEqualizeHistEnabled = False

    # 初始化/刷新数据库
    def initDb(self):
        # 刷新前重置tableWidget
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
        try:
            if not os.path.isfile(self.database):
                raise FileNotFoundError

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            res = cursor.execute('SELECT * FROM users')
            for row_index, row_data in enumerate(res):
                self.tableWidget.insertRow(row_index)
                for col_index, col_data in enumerate(row_data):
                    self.tableWidget.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except FileNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.isDbReady = False
            self.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except Exception:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            self.initdb.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化/刷新数据库失败')
        else:
            cursor.close()
            conn.close()

            self.UserlcdNumber.display(dbUserCount)
            if not self.isDbReady:
                self.isDbReady = True
                self.logQueue.put('Success：数据库初始化完成，发现用户数：{}'.format(dbUserCount))
                self.initdb.setText('刷新数据库')
                self.initdb.setIcon(QIcon('./icons/success.png'))
                self.trainButton.setToolTip('')
                self.trainButton.setEnabled(True)
                self.QueryButton.setToolTip('')
                self.QueryButton.setEnabled(True)
            else:
                self.logQueue.put('Success：刷新数据库成功，发现用户数：{}'.format(dbUserCount))

    # 查询用户
    def queryUser(self):
        stu_id = self.QuerylineEdit.text().strip()
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
            ret = cursor.fetchall()
            if not ret:
                raise RecordNotFound
            face_id = ret[0][1]
            cn_name = ret[0][2]
        except RecordNotFound:
            self.QueryButton.setIcon(QIcon('./icons/error.png'))
            self.queryResultLabel.setText('<font color=red>Error：此用户不存在</font>')
        except Exception as e:
            logging.error('读取数据库异常，无法查询到{}的用户信息'.format(stu_id))
            self.queryResultLabel.clear()
            self.QueryButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，查询失败')
        else:
            self.queryResultLabel.clear()
            self.QueryButton.setIcon(QIcon('./icons/success.png'))
            self.UsernumlineEdit.setText(stu_id)
            self.UsernamelineEdit.setText(cn_name)
            self.FaceIDLineedit.setText(str(face_id))
            self.deleteUserButton.setEnabled(True)
        finally:
            cursor.close()
            conn.close()

    # 删除用户
    def deleteUser(self):
        text = '从数据库中删除该用户，同时删除相应人脸数据，<font color=red>该操作不可逆！</font>'
        informativeText = '<b>是否继续？</b>'
        ret = DataManageUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)

        if ret == QMessageBox.Yes:
            stu_id = self.UsernumlineEdit.text()
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('DELETE FROM users WHERE stu_id=?', (stu_id,))
            except Exception as e:
                cursor.close()
                logging.error('无法从数据库中删除{}'.format(stu_id))
                self.deleteUserButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，删除失败')
            else:
                cursor.close()
                conn.commit()
                if os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):
                    try:
                        shutil.rmtree('{}/stu_{}'.format(self.datasets, stu_id))
                    except Exception as e:
                        logging.error('系统无法删除删除{}/stu_{}'.format(self.datasets, stu_id))
                        self.logQueue.put('Error：删除人脸数据失败，请手动删除{}/stu_{}目录'.format(self.datasets, stu_id))

                text = '你已成功删除学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                informativeText = '<b>请在右侧菜单重新训练人脸数据。</b>'
                DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                self.UsernumlineEdit.clear()
                self.UsernamelineEdit.clear()
                self.FaceIDLineedit.clear()
                self.initDb()
                self.deleteUserButton.setIcon(QIcon('./icons/success.png'))
                self.deleteUserButton.setEnabled(False)
                self.QueryButton.setIcon(QIcon())
            finally:
                conn.close()

    # 检测人脸
    def detectFace(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.isEqualizeHistEnabled:
            gray = cv2.equalizeHist(gray)      #将图片直方图均衡化
        #face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml') #加载Face Casecade或haarcascade文件来预测图像中的人脸
        #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(90, 90))
        self.faceCascade = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.faceCascade.setInput(blob)
        faces = self.faceCascade.forward()
        h, w, c = img.shape
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.6:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        if (len(faces) == 0):
            return None, None
        #(x, y, w, h) = faces[0]
        return gray[startY:endY, startX:endX], faces[0]

    # 准备图片数据
    def prepareTrainingData(self, data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []

        face_id = 1
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        # 遍历人脸库
        for dir_name in dirs:
            if not dir_name.startswith('stu_'):
                continue
            stu_id = dir_name.replace('stu_', '')
            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                ret = cursor.fetchall()         #返回多个元组，即返回多条记录(rows),如果没有结果,则返回 ()
                if not ret:
                    raise RecordNotFound
                cursor.execute('UPDATE users SET face_id=? WHERE stu_id=?', (face_id, stu_id,))
            except RecordNotFound:
                logging.warning('数据库中找不到学号为{}的用户记录'.format(stu_id))
                self.logQueue.put('发现学号为{}的人脸数据，但数据库中找不到相应记录，已忽略'.format(stu_id))
                continue
            subject_dir_path = data_folder_path + '/' + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            for image_name in subject_images_names:
                if image_name.startswith('.'):
                    continue
                image_path = subject_dir_path + '/' + image_name
                image = cv2.imread(image_path)    #读取图片信息
                face, rect = self.detectFace(image)  #人脸检测
                if face is not None:
                    faces.append(face)
                    labels.append(face_id)
            face_id = face_id + 1

        cursor.close()
        conn.commit()
        conn.close()

        return faces, labels

    # 训练人脸数据
    def train(self):
        try:
            if not os.path.isdir(self.datasets):
                raise FileNotFoundError

            text = '系统将开始训练人脸数据，界面会暂停响应一段时间，完成后会弹出提示。'
            informativeText = '<b>训练过程请勿进行其它操作，是否继续？</b>'
            ret = DataManageUI.callDialog(QMessageBox.Question, text, informativeText,
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            if ret == QMessageBox.Yes:
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()   #用名为cv2.face.LBPHFaceRecognizer_create（）的函数来实现识别器
                #创建LBPH识别器来时训练，也可以选择Eigen或者Fisher识别器
                #face_recognizer = cv2.face.FisherFaceRecognizer_create()
                if not os.path.exists('./recognizer'):
                    os.makedirs('./recognizer')
            faces, labels = self.prepareTrainingData(self.datasets)
            face_recognizer.train(faces, np.array(labels))    #为给定的带有标签id的图像设置训练计算机
            face_recognizer.save('./recognizer/trainingData.yml')  #将训练过的数据保存到.yml文件中，以便识别
        except FileNotFoundError:
            logging.error('系统找不到人脸数据目录{}'.format(self.datasets))
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('未发现人脸数据目录{}，你可能未进行人脸采集'.format(self.datasets))
        except Exception as e:
            logging.error('遍历人脸库出现异常，训练人脸数据失败')
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：遍历人脸库出现异常，训练失败')
        else:
            text = '<font color=green><b>Success!</b></font> 系统已生成./recognizer/trainingData.yml'
            informativeText = '<b>人脸数据训练完成！</b>'
            DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
            self.trainButton.setIcon(QIcon('./icons/success.png'))
            self.logQueue.put('Success：人脸数据训练完成')
            self.initDb()

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
        msg.setWindowTitle('OpenCV Face Recognition System - DataManage')
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

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = DataManageUI()
    window.show()
    sys.exit(app.exec())
