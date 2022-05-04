import sys, cv2, time

from gui.x import Ui_TabWidget

from PyQt5 import QtGui, QtWidgets

from PyQt5.QtWidgets import QFileDialog,QTabWidget

from PyQt5.QtCore import QThread, pyqtSignal, Qt

from PyQt5.QtGui import QPixmap, QImage


class mywindow(QTabWidget,Ui_TabWidget): #这个窗口继承了用QtDesignner 绘制的窗口

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)

    def videoprocessing(self):
        print("gogo")
        global videoName #在这里设置全局变量以便在线程中使用
        videoName,videoType= QFileDialog.getOpenFileName(self,
                                    "打开视频",
                                    "",
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.mp4;;*.avi;;All Files (*)")
        #cap = cv2.VideoCapture(str(videoName))
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    def imageprocessing(self):
        print("hehe")
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    "",
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")

        #利用qlabel显示图片
        print(str(imgName))
        png = QtGui.QPixmap(imgName).scaled(self.label_2.width(), self.label_2.height())#适应设计label时的大小
        self.label_2.setPixmap(png)

class Thread(QThread):#采用线程来播放视频

    changePixmap = pyqtSignal(QtGui.QImage)
    def run(self):
        cap = cv2.VideoCapture(videoName)
        print(videoName)
        while (cap.isOpened()==True):
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.01) #控制视频播放的速度
            else:
                break


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())