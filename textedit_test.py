from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 600)

        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")


        self.verticalLayout.addLayout(self.horizontalLayout)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.clearButton = QtWidgets.QPushButton(self)
        self.clearButton.move(100, 100)
        self.clearButton.setObjectName("clearButton")
        self.clearButton.setMaximumSize(QtCore.QSize(16, 16))
        self.clearButton.setStyleSheet('''QPushButton
                   {
                   font-family:"Webdings";
                   background:#C3C3C3;border-radius:8px;
                   border:none;
                   font-size:10px;
                   }
                   QPushButton:hover{background:#E6E6E6;}''')


        #self.textEdit.setGeometry(50, 50, 100, 100)
        self.retranslateUi(Form)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

        self.clearButton.setToolTip(_translate("Form", "<html><head/><body><p>清空</p></body></html>"))
        self.clearButton.setText(_translate("Form", "r"))
