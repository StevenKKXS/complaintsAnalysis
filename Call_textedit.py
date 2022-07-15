#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：mgboy time:2020/7/26
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from textedit_test import Ui_Form
from CallTitletest import TitleWindow
import time


class MyWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.textEdit.move(20, 20)
        # textstyle =
        # self.textEdit.setStyleSheet()
        hbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, self.textEdit)
        vbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical, self.textEdit)
        hstyle = '''QScrollBar
                           {
                           background : white;
                           }


                           QScrollBar::handle
                           {
                               background:gray;
                               border-radius:4px;
                               margin-top:4px;
                               margin-bottom:4px;
                               margin-left:4px;
                               margin-right:120px;  
                           }
                   
                           QScrollBar::add-line
                           {
                               height:0px;
                               width:0px;
                           }
                           QScrollBar::sub-line
                           {
                               height:0px;
                               width:0px;
                           }
                           '''
        vstyle = '''QScrollBar
                           {
                           background : white;
                           }


                           QScrollBar::handle
                           {
                               background:gray;
                               border-radius:4px;
                               margin-left:0px;
                               margin-right:8px; 
                               margin-top:25px;
                               margin-bottom:45px; 
                           }
                   
                           QScrollBar::add-line
                           {
                               height:0px;
                               width:0px;
                           }
                           QScrollBar::sub-line
                           {
                               height:0px;
                               width:0px;
                           }
                           '''
        hbar.setStyleSheet(hstyle
                           )
        vbar.setStyleSheet(vstyle)
        self.textEdit.setHorizontalScrollBar(hbar)
        self.textEdit.setVerticalScrollBar(vbar)
        self.textEdit.setObjectName("textEdit")

        self.textEdit.setViewportMargins(20, 20, 20, 20)

        self.verticalLayout.addWidget(self.textEdit)

        self.textEdit.setPlaceholderText('在这里输入描述')

        # self.textEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.action1 = QtWidgets.QAction(self.textEdit)
        self.action1.setIcon(QApplication.style().standardIcon(1))
        self.action1.triggered.connect(lambda: self.selectType(self.selectApply))

        self.predictButton = QtWidgets.QPushButton(self)
        self.predictButton.move(200, 200)
        self.predictButton.setObjectName('predictButton')
        self.predictButton.setText('确认')

        self.model = None

        # self.verticalLayout.addWidget(self.appLinkEdit)

        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setObjectName('textBrowser')
        self.verticalLayout.addWidget(self.textBrowser)
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 100))
        self.textBrowser.setViewportMargins(20, 20, 20, 20)

        # self.verticalLayout.
        # self.textBrowser.setStyleSheet("border-image:url(background3.png);padding:10px 10px")
        self.clearButton.clicked.connect(self.clear)
        self.predictButton.clicked.connect(self.predict)

    def clear(self):
        self.textEdit.clear()

    def resizeEvent(self, QResizeEvent):
        self.predictButton.move(self.width() - 120, int(self.height() - 157))
        self.clearButton.move(self.width() - 31, 15)

    def load_model(self):
        from predict import LSTM_model
        self.model = LSTM_model()

    def predict(self):
        self.textBrowser.setText('请稍后')
        if self.model == None:
            self.load_model()
            print(233)
        txt = self.textEdit.toPlainText()
        res = self.model.predict(txt)
        self.textBrowser.setText(res)


if __name__ == "__main__":
    # 适配2k等高分辨率屏幕
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    content = MyWindow()
    myWin = TitleWindow(widget_2_sub=content, icon_path=None, title='客户投诉分类')
    myWin.show()

    sys.exit(app.exec_())
