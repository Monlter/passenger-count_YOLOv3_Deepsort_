# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test1.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(941, 777)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(60, 60, 121, 511))
        self.widget.setObjectName("widget")
        self.Title3 = QtWidgets.QLabel(self.widget)
        self.Title3.setGeometry(QtCore.QRect(10, 340, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(10)
        self.Title3.setFont(font)
        self.Title3.setObjectName("Title3")
        self.Title2 = QtWidgets.QLabel(self.widget)
        self.Title2.setGeometry(QtCore.QRect(10, 190, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(10)
        self.Title2.setFont(font)
        self.Title2.setObjectName("Title2")
        self.Title1 = QtWidgets.QLabel(self.widget)
        self.Title1.setGeometry(QtCore.QRect(10, 25, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(10)
        self.Title1.setFont(font)
        self.Title1.setObjectName("Title1")
        self.lcdNumber = QtWidgets.QLCDNumber(self.widget)
        self.lcdNumber.setGeometry(QtCore.QRect(10, 70, 101, 71))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber_2 = QtWidgets.QLCDNumber(self.widget)
        self.lcdNumber_2.setGeometry(QtCore.QRect(10, 230, 101, 71))
        self.lcdNumber_2.setObjectName("lcdNumber_2")
        self.lcdNumber_3 = QtWidgets.QLCDNumber(self.widget)
        self.lcdNumber_3.setGeometry(QtCore.QRect(10, 380, 101, 71))
        self.lcdNumber_3.setObjectName("lcdNumber_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 610, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.TitleAll = QtWidgets.QLabel(self.centralwidget)
        self.TitleAll.setGeometry(QtCore.QRect(250, 0, 411, 51))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.TitleAll.setFont(font)
        self.TitleAll.setObjectName("TitleAll")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(610, 650, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(130, 650, 461, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(790, 650, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.videoPlay = QtWidgets.QWidget(self.centralwidget)
        self.videoPlay.setGeometry(QtCore.QRect(200, 70, 671, 511))
        self.videoPlay.setObjectName("videoPlay")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 650, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 941, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title3.setText(_translate("MainWindow", "FPS"))
        self.Title2.setText(_translate("MainWindow", "总人数"))
        self.Title1.setText(_translate("MainWindow", "当前人流量"))
        self.label.setText(_translate("MainWindow", "请输入需要检测的视频："))
        self.TitleAll.setText(_translate("MainWindow", "基于图像处理的行人目标检测系统"))
        self.pushButton.setText(_translate("MainWindow", "离线"))
        self.pushButton_3.setText(_translate("MainWindow", "检测"))
        self.pushButton_2.setText(_translate("MainWindow", "实时"))
