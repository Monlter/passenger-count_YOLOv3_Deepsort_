import sys
from PyQt5.QtWidgets import QApplication,QFileDialog,QMainWindow
from gui.GUI import Ui_MainWindow
from gui import main_gui
from yolo import YOLO
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.ConfigProto()
config.gpu_options.allow_growth= True #不全部占满显存, 按需分配
session = tf.Session(config=config)



class DetailUI(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(DetailUI,self).__init__()
        self.fname = ''
        self.setupUi(self)
        self.setWindowTitle('基于图像处理的行人识别')
        # self.player = QMediaPlayer()
        # self.player.setVideoOutput(self.videoPlay)

    def OpenVideo(self):
        try:
            self.fname , _ = QFileDialog.getOpenFileName(self,'open file',"/","Video files(*.mp4 *.avi)")
            self.textBrowser.setText(self.fname)
        except:
            self.textBrowser.setText("打开文件失败，可能是文件内型错误")

    def OpenCamera(self):
        self.fname = 0
        self.textBrowser.setText('您当前打开的是摄像头')


    def runModel(self):
        print(self.fname)
        print(self)
        main_gui.main(YOLO(), self.fname, self)

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = DetailUI()
   ex.show()
   sys.exit(app.exec_())