import DataManage
import DataRecord
import Face_recognition
import sys
import facerecoDataRecord
import facereco
import facedatamanage

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtCore,QtGui
class MainWindowUI(QMainWindow):
    def __init__(self):       #__init__方法
        super(MainWindowUI, self).__init__()  #构造函数初始化
        loadUi('./ui/mainWindow.ui', self) #下载UI界面
        self.setWindowIcon(QIcon('./icons/icon.png'))  #设置图标
        self.FaceDetectButton.clicked.connect(self.faceDetect)
        self.TrainButton.clicked.connect(self.Train)
        self.FacerecognizeButton.clicked.connect(self.facerecognition)
        self.Facereco.clicked.connect(self.facerecogition2)
        self.Informationrecord.clicked.connect(self.Inforecord)
        self.Qureymanage.clicked.connect(self.datamanage)
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
    def faceDetect(self):
        self.faceDectUI=DataRecord.DataRecordUI()
        self.faceDectUI.show()
    def Train(self):
        self.TrainUI=DataManage.DataManageUI()
        self.TrainUI.show()
    def facerecognition(self):
        self.facerecoUI=Face_recognition.CoreUI()
        self.facerecoUI.show()
    def facerecogition2(self):
        self.facerecoUI2=facereco.FaceCoreUI()
        self.facerecoUI2.show()
    def Inforecord(self):
        self.inforecordUI=facerecoDataRecord.FaceDataRecordUI()
        self.inforecordUI.show()
    def datamanage(self):
        self.datamanageUI=facedatamanage.FaceDataManageUI()
        self.datamanageUI.show()

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
    app = QApplication(sys.argv)
    window =MainWindowUI()
    window.show()
    sys.exit(app.exec())






