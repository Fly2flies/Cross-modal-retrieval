"""
重写一个带点击事件的Label
"""
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot,pyqtSignal

class MyLabel(QtWidgets.QLabel):
    signalClicked = pyqtSignal()  # 点击信号
    def __init__(self,parent = None):
        super(MyLabel,self).__init__(parent)
        self.isClicked = False # 没有鼠标点击
    def mousePressEvent(self, e):
        self.isClicked = True
    def mouseMoveEvent(self, e): # 移开鼠标表示不确定
        self.isClicked = False
    def mouseReleaseEvent(self, e):
        if self.isClicked:
            self.isClicked = False
            self.signalClicked.emit()

