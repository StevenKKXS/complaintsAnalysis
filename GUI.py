import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton, QApplication)
from PyQt5 import QtCore, QtGui, QtWidgets


class basicWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 获取屏幕分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()

        # 自适应屏幕分辨率
        resize = (self.screenwidth ** 2 + self.screenheight ** 2) ** 0.5 / 2200
        self.resize(int(800 * resize), int(600 * resize))
        self.setStyleSheet('''QWidget{background-color:rgb(247,247,247);}''')
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setObjectName("textEdit")

        self.textEdit.setStyleSheet(
            "border:2px groove gray;border-radius:10px;background-color: white;padding:10px 10px")

        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.textEdit.setPlaceholderText('在这里输入描述')
        # grid_layout.addWidget(self.textEditBackground,0,0)
        grid_layout.addWidget(self.textEdit, 0, 0)
        self.textBrowser = QtWidgets.QTextBrowser()
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setStyleSheet(
            "border:2px groove gray;border-radius:10px;background-color: white;padding:10px 10px")
        grid_layout.addWidget(self.textBrowser, 1, 0)
        grid_layout.setRowStretch(0, 5)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setSpacing(50)

        self.pushButton = QtWidgets.QPushButton()
        grid_layout.addWidget(self.pushButton, 2, 0)
        self.pushButton.clicked.connect(self.button_clicked)
        '''
        for x in range(3):
            for y in range(3):
                button = QPushButton(str(str(3 * x + y)))

                grid_layout.addWidget(button, x, y)

            grid_layout.setColumnStretch(x, x + 1)
        
        self.setWindowTitle('Basic Grid Layout')
        '''

    def button_clicked(self):
        txt = self.textEdit.toPlainText()
        res = self.lstm_model.predict(txt)
        self.textBrowser.setText(res)

    def load_lstm_model(self):
        from predict import LSTM_model
        self.lstm_model = LSTM_model()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    windowExample = basicWindow()
    windowExample.show()
    windowExample.load_lstm_model()
    sys.exit(app.exec_())
