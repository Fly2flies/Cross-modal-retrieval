# -*- coding: utf-8 -*-
"""界面设计"""
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtGui import QCursor,QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy

from QueryApp.MyButton import SwitchButton
from QueryApp.MyLabel import MyLabel

import sys
import os
import qtawesome

class QueryUi(QtWidgets.QMainWindow):
    def __init__(self):
        super(QueryUi, self).__init__()
        self.m_flag = False # 是否按住窗口进行移动
        self.FullWindow = False # 是否全屏
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(1120, 600)
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 25, 4)
        self.main_layout.addWidget(self.right_widget, 0, 4, 25, 16)
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        ##### 左侧组件 #####
        ## 头部按钮
        self.left_top_button_widget = QtWidgets.QWidget()
        self.left_top_button_widget.setObjectName('left_top_button')
        self.left_top_button_layout = QtWidgets.QGridLayout()
        self.left_top_button_widget.setLayout(self.left_top_button_layout)

        self.left_close = QtWidgets.QPushButton("X")  # 关闭按钮  第二个
        self.left_visit = QtWidgets.QPushButton("O")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("-")  # 最小化按钮 第一个
        self.left_close.clicked.connect(self.close)
        self.left_mini.clicked.connect(self.showMinimized)
        self.left_visit.clicked.connect(self.changeWindow)
        # 美化
        self.left_close.setFixedSize(15, 15)
        self.left_visit.setFixedSize(15, 15)
        self.left_mini.setFixedSize(15, 15)

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        # 合并按钮
        self.left_top_button_layout.addWidget(self.left_mini,0,0,1,1)
        self.left_top_button_layout.addWidget(self.left_visit,0,1,1,1)
        self.left_top_button_layout.addWidget(self.left_close,0,2,1,1)

        self.left_label_1 = QtWidgets.QPushButton("查询图片")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("执行操作")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("程序初始化")
        self.left_label_3.setObjectName('left_label')
        self.left_label_4 = QtWidgets.QPushButton("数据初始化")
        self.left_label_4.setObjectName('left_label')

        self.left_label_5 = QtWidgets.QLabel("一键初始化")
        self.left_label_5.setObjectName("gpu_label")
        self.left_label_5.setFont(QFont("Microsoft Yahei", 10))

        self.left_label_5.setAlignment(Qt.AlignLeft)
        self.left_label_5.setStyleSheet("color:blue")
        self.left_gpu_widget = QtWidgets.QWidget()
        self.left_button_9 = SwitchButton()
        # 添加GPU选项

        self.left_gpu_widget.setObjectName("Init_Data")
        self.left_gpu_widget_layout = QtWidgets.QHBoxLayout()
        self.left_gpu_widget.setLayout(self.left_gpu_widget_layout)
        self.left_gpu_widget_layout.addWidget(self.left_label_5)
        self.left_gpu_widget_layout.addWidget(self.left_button_9)

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.photo', color='black'), "打开图片")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.search', color='black'), "开始检索")
        self.left_button_2.setObjectName('left_button')

        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "加载图像模型")
        self.left_button_3.setObjectName('left_button')

        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "加载文本模型")
        self.left_button_4.setObjectName('left_button')

        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "选择图片路径")
        self.left_button_5.setObjectName('left_button')

        self.left_button_6_1 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "文本数据")
        self.left_button_6_2 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "图像数据")
        self.left_button_6_1.setObjectName('left_button')
        self.left_button_6_2.setObjectName('left_button')

        self.left_button_7 = QtWidgets.QPushButton(qtawesome.icon('fa.cog', color='black'), "选择字典")
        self.left_button_7.setObjectName('left_button')

        self.left_button_init_data = QtWidgets.QPushButton("一键初始化")
        self.left_button_init_data.setObjectName("left_button")

        self.left_button_8 = QtWidgets.QLabel()
        self.left_button_8.setObjectName('left_button')
        # 左边按钮布局
        self.left_layout.addWidget(self.left_top_button_widget,0,0,1,3)

        self.left_layout.addWidget(self.left_label_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_8, 3, 0, 11, 4)

        self.left_layout.addWidget(self.left_label_2, 14, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 15, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 16, 0, 1, 3)

        self.left_layout.addWidget(self.left_label_3, 17, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 18, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 19, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 20, 0, 1, 3)

        self.left_data_widget = QtWidgets.QWidget()
        self.left_data_widget_layout = QtWidgets.QHBoxLayout()
        self.left_data_widget.setLayout(self.left_data_widget_layout)
        self.left_data_widget_layout.addWidget(self.left_button_6_1,0)
        self.left_data_widget_layout.addWidget(self.left_button_6_2)

        self.left_layout.addWidget(self.left_data_widget, 21, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_7, 22, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_4, 23, 0, 1, 3)
        self.left_layout.addWidget(self.left_gpu_widget,24,0,1,3)
        # 左侧美化
        self.left_widget.setStyleSheet('''
            QWidget#left_widget{
                background:gray;
                border - top: 1px solid white;
                border - bottom: 1px solid white;
                border - left: 1px solid white;
                border - top - left - radius: 10px;
                border - bottom - left - radius: 10px;
            }
            QToolButton#left_button{
                border:none;
                text-align: center;   
            }
            QToolButton:hover{border-bottom:2px solid #F76677;}
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QLabel#copyright_label{
                border:none;
                color: white;
                text-align:right;
                font-size:14px;
                font-weight:400;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button{
                font-size:14px;
                font-weight:700;
                font-family:"微软雅黑";
            }
        ''')

        ###### 右侧内容 ######
        self.right_bar_widget = QtWidgets.QWidget()  # 右侧顶部搜索框部件
        self.right_bar_layout = QtWidgets.QGridLayout()  # 右侧顶部搜索框网格布局
        self.right_bar_widget.setLayout(self.right_bar_layout)
        # 搜索图标
        self.search_icon = MyLabel(chr(0xf002) + ' ' + 'Search')
        self.search_icon.setFont(qtawesome.font('fa', 16))
        # 搜索框
        self.right_bar_widget_search_input = QtWidgets.QLineEdit()
        self.right_bar_widget_search_input.setPlaceholderText("Please input Image Captions to search !")
        self.right_bar_layout.addWidget(self.search_icon, 0, 0, 1, 1)
        self.right_bar_layout.addWidget(self.right_bar_widget_search_input, 0, 1, 1, 15)
        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 2, 16)

        # 热搜关键字
        self.right_recommend_label = MyLabel("Hot KeyWords " + chr(0xf021))
        self.right_recommend_label.setObjectName('right_lable')
        self.right_recommend_label.setFont(qtawesome.font('fa',16))

        # 推荐关键词的label
        self.right_recommend_widget = QtWidgets.QWidget()
        self.right_recommend_layout = QtWidgets.QGridLayout()
        self.right_recommend_widget.setLayout(self.right_recommend_layout)

        self.right_captions_line1 = QtWidgets.QLineEdit("Caption1")
        self.right_captions_line1.setObjectName("caption_label")
        self.right_captions_line2 = QtWidgets.QLineEdit("Caption2")
        self.right_captions_line2.setObjectName("caption_label")
        self.right_captions_line3 = QtWidgets.QLineEdit("Caption3")
        self.right_captions_line3.setObjectName("caption_label")

        self.right_captions_score1 = QtWidgets.QLabel("Simaliry")
        self.right_captions_score2 = QtWidgets.QLabel("Simaliry")
        self.right_captions_score3 = QtWidgets.QLabel("Simaliry")

        self.right_captions_socres = [self.right_captions_score1, self.right_captions_score2,
                                      self.right_captions_score3]
        self.right_captions_lines = [self.right_captions_line1, self.right_captions_line2,
                                    self.right_captions_line3]

        for i,caption_label in enumerate(self.right_captions_lines):
            caption_label.setReadOnly(True)
            caption_label.setStyleSheet("""
                QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
                }
            """)
            self.right_recommend_layout.addWidget(caption_label,i,0,1,10)
            self.right_recommend_layout.addWidget(self.right_captions_socres[i],i,10,1,1)

        self.right_layout.addWidget(self.right_recommend_label,2,0,1,16)
        self.right_layout.addWidget(self.right_recommend_widget,3,0,3,16)

        # 搜索的结果展示
        self.right_result_label = QtWidgets.QLabel("Serach Image Results" + chr(0xf0a7))
        self.right_result_label.setFont(qtawesome.font('fa',16))
        self.right_result_label.setObjectName('right_lable')
        self.right_layout.addWidget(self.right_result_label, 6, 0, 1, 16)

        self.resultlBt1 = QtWidgets.QToolButton()
        self.resultlBt2 = QtWidgets.QToolButton()
        self.resultlBt3 = QtWidgets.QToolButton()
        self.resultlBt4 = QtWidgets.QToolButton()
        self.resultlBt5 = QtWidgets.QToolButton()
        self.resultlBt6 = QtWidgets.QToolButton()
        self.resultlBt7 = QtWidgets.QToolButton()
        self.resultlBt8 = QtWidgets.QToolButton()
        self.results = [self.resultlBt1,self.resultlBt2,self.resultlBt3,self.resultlBt4,
                        self.resultlBt5,self.resultlBt6,self.resultlBt7,self.resultlBt8]
        for i,resultBt in enumerate(self.results):
            resultBt.setIconSize(QtCore.QSize(224, 224))
            resultBt.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        # 保存显示结果
        for resultBt in self.results:
            resultBt.setContextMenuPolicy(Qt.CustomContextMenu)
        # 显示结果布局
        self.right_resultlist_widget = QtWidgets.QWidget()
        self.right_resultlist_layout = QtWidgets.QGridLayout()
        self.right_resultlist_widget.setLayout(self.right_resultlist_layout)
        for i,resultBt in enumerate(self.results):
            self.right_resultlist_layout.addWidget(resultBt,i // 4,i % 4)
        self.right_layout.addWidget(self.right_resultlist_widget,7,0,14,16)

        self.pageDownBt = QtWidgets.QPushButton(qtawesome.icon('fa.arrow-circle-left', color='black'), "Last")
        self.pageDownBt.setObjectName('left_button')
        self.pageUpBt = QtWidgets.QPushButton(qtawesome.icon('fa.arrow-circle-right', color='black'), "Next")
        self.pageUpBt.setObjectName('left_button')

        self.right_layout.addWidget(self.pageUpBt,21, 10, 2, 2)
        self.right_layout.addWidget(self.pageDownBt, 21, 4, 2, 2)

        # 搜索框美化
        self.right_bar_widget_search_input.setStyleSheet('''
        QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
            }''')
        self.right_resultlist_widget.setStyleSheet('''
                QToolButton{border:none;}
                QToolButton:hover{border-bottom:2px solid #F76677;}
            ''')
        # 右侧美化
        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QPushButton#commend_label{
                border:none;
                border-bottom:1px solid white;
                font-size:14px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton{border:none;}
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
            }
        ''')
        # 窗口美化
        self.setWindowOpacity(0.99)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.main_layout.setSpacing(0)

    def changeWindow(self):
        if self.FullWindow is False:
            self.FullWindow = True
            self.showFullScreen()
        else:
            self.FullWindow = False
            self.showNormal()
    # 鼠标左键控制移动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = QueryUi()
    gui.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()

