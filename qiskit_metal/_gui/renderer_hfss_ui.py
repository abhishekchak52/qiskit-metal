# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Temp\GitHub\qiskit-metal\qiskit_metal\_gui\renderer_hfss_ui.ui'
#
# Created: Sat Jun 19 22:02:29 2021
#      by: pyside6-uic  running on PySide2 5.13.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PySide6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(646, 544)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.instructionsLabel = QtWidgets.QLabel(self.centralwidget)
        self.instructionsLabel.setObjectName("instructionsLabel")
        self.gridLayout.addWidget(self.instructionsLabel, 0, 0, 1, 1)
        self.instructionsLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.instructionsLabel_2.setAlignment(QtCore.Qt.AlignCenter)
        self.instructionsLabel_2.setObjectName("instructionsLabel_2")
        self.gridLayout.addWidget(self.instructionsLabel_2, 0, 1, 1, 1)
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setObjectName("listView")
        self.gridLayout.addWidget(self.listView, 1, 0, 3, 1)
        self.treeView = QTreeView_Base(self.centralwidget)
        self.treeView.setRootIsDecorated(False)
        self.treeView.setObjectName("treeView")
        self.gridLayout.addWidget(self.treeView, 1, 1, 1, 1)
        self.solutionLabel = QtWidgets.QLabel(self.centralwidget)
        self.solutionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.solutionLabel.setObjectName("solutionLabel")
        self.gridLayout.addWidget(self.solutionLabel, 2, 1, 1, 1)
        self.solnComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.solnComboBox.setObjectName("solnComboBox")
        self.solnComboBox.addItem("")
        self.solnComboBox.addItem("")
        self.gridLayout.addWidget(self.solnComboBox, 3, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.refreshButton = QtWidgets.QPushButton(self.centralwidget)
        self.refreshButton.setObjectName("refreshButton")
        self.horizontalLayout_4.addWidget(self.refreshButton)
        self.selectAllButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectAllButton.setObjectName("selectAllButton")
        self.horizontalLayout_4.addWidget(self.selectAllButton)
        self.deselectAllButton = QtWidgets.QPushButton(self.centralwidget)
        self.deselectAllButton.setObjectName("deselectAllButton")
        self.horizontalLayout_4.addWidget(self.deselectAllButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 4, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.confirmButton = QtWidgets.QPushButton(self.centralwidget)
        self.confirmButton.setObjectName("confirmButton")
        self.horizontalLayout.addWidget(self.confirmButton)
        self.gridLayout.addLayout(self.horizontalLayout, 4, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 646, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.selectAllButton.clicked.connect(
            MainWindow.select_all)  # type: ignore
        self.deselectAllButton.clicked.connect(
            MainWindow.deselect_all)  # type: ignore
        self.refreshButton.clicked.connect(MainWindow.refresh)  # type: ignore
        self.confirmButton.clicked.connect(
            MainWindow.choose_checked_components)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HFSS Renderer"))
        self.instructionsLabel.setText(
            _translate("MainWindow", "Check off components to export:"))
        self.instructionsLabel_2.setText(
            _translate("MainWindow", "Renderer options"))
        self.solutionLabel.setText(_translate("MainWindow", "Solution type:"))
        self.solnComboBox.setItemText(0, _translate("MainWindow", "Eigenmode"))
        self.solnComboBox.setItemText(1, _translate("MainWindow",
                                                    "Driven Modal"))
        self.refreshButton.setText(_translate("MainWindow", "Refresh List"))
        self.selectAllButton.setText(_translate("MainWindow", "Select All"))
        self.deselectAllButton.setText(_translate("MainWindow", "Deselect All"))
        self.confirmButton.setText(_translate("MainWindow",
                                              "Confirm Selection"))


from . import main_window_rc_rc
from .tree_view_base import QTreeView_Base
