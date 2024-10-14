import sys
from PyQt5.QtWidgets import *
import time
import matplotlib
import matplotlib as mpl
import shutil
import os
import numpy as np
import matplotlib.animation as animation
import math
import computerview
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCa
from matplotlib.figure import Figure
from mainwindow import Ui_MainWindow
from resultwindow import Ui_resultWindow

matplotlib.use("QT5Agg")
# 数据库的创建和初始赋值
computerview.make_DB_reuslt()
computerview.make_DB_loadmark()
computerview.pre_Mark()

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class ResultFigure(FigureCa):
    def __init__(self,width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(ResultFigure, self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

    def remind(self):
        remindBox = QMessageBox()
        remindBox.about(None,"开始","以此视频开始测试")

    def error(self):
        errorBox = QMessageBox()
        errorBox.critical(None, '错误', '请先选择一个视频文件')

class resultWindow(QMainWindow, Ui_resultWindow):

    def __init__(self):
        super(resultWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("结果显示")
        # 图像的实例化
        self.F = ResultFigure(width=3, height=2, dpi=100)
        self.gridlayout = QGridLayout(self.imageBox)
        self.gridlayout.addWidget(self.F)
    def plottrail(self, x, y):
        self.x = x
        self.p = 0.0
        
        self.y = y
        self.line, = self.F.axes.plot(self.x, self.y, animated=True, lw=2)
        self.ani = animation.FuncAnimation(self.F.figure, self.update_line,
                                           blit=True, interval=25)
    def update_line(self, i):
        x = self.x
        y = self.y
        self.line.set_data(x, y)
        return [self.line]


    def on_stop(self):
        self.ani._stop()
    '''
    def plotcos(self, t, s):
        self.F.axes.plot(t, s)
        self.F.fig.suptitle("路径")
    '''
    def finish(self):
        finishBox = QMessageBox()
        finishBox.about(None, "提示", "本次识别已结束")

def fuction():
    MainWindow.fileSelcet.clicked.connect(fileDefine)
    MainWindow.fileEdit.selectionChanged.connect(fileDefine)
    MainWindow.proStart.clicked.connect(pro_Start)


def fileDefine():
    filepath = QFileDialog.getOpenFileName(None, "选择一个视频文件", filter="视频文件(*.mp4)")
    if filepath[0] == '':
        MainWindow.fileEdit.setText("请选择想要检测的视频")
    else:
        MainWindow.fileEdit.setText(filepath[0])

def pro_Start():
    filepath = MainWindow.fileEdit.text()
    if filepath == "请选择想要检测的视频":
        MainWindow.error()
        MainWindow.setupUi(MainWindow)
        fuction()
        return -1
    MainWindow.remind()
    print(filepath)
    result = computerview.Video2Image(filepath)
    # 绘制路径，显示当前的位置信息(并不实时，所以移到computview.py中
    Result = resultWindow()
    MainWindow.gridLayout2.addWidget(Result)
    X = []
    Y = []
    dX = []
    dY = []
    for i in range(0, len(result)):
        def distance(point1, point2):
            return pow(pow(float(point1[0]) - float(point2[0]), 2) + pow(float(point1[1]) - float(point2[1]), 2), 0.5)
        if result[i][0] == 0:
            Result.resultNowbox.setText("未检测到路标")
            Result.show()
            QApplication.processEvents()
            time.sleep(0.5)
        else:
            Result.resultNowbox.setText("mark_id = %s\ndX = %s\ndY = %s\nnowangle = %s\n" % (result[i][0],result[i][1],result[i][2],result[i][3]))
            X.append(result[i][1])
            Y.append(result[i][2])
            Result.plottrail(X, Y)
            Result.show()
            QApplication.processEvents()
            time.sleep(0.5)
    # 实现数据库的存储，实在太慢了，只好单独拎出来
    for i in result[:]:
        computerview.save_process(i[0], i[1], i[2], i[3])
    Result.finish()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    MainWindow = MainWindow()
    MainWindow.show()
    fuction()

    sys.exit(app.exec_())
