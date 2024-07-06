import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import cv2

class CamSettings_Widget(QWidget):
    def __init__(self):
        super(CamSettings_Widget, self).__init__()
        #self.resize(1000, 1000)

        self.setWindowTitle("Camera Settings Panel - GyroSpin App")

        #---   Layout total ---#
        self.LayoutTotal = QVBoxLayout()


        self.HBL1 = QHBoxLayout()
        self.LabelScale1 = QLabel('Cam1 Scale (1px = ??cm) " ')
        self.EditLineScale1 = QLineEdit()
        self.HBL1.addWidget(self.LabelScale1)
        self.HBL1.addWidget(self.EditLineScale1)
        self.LayoutTotal.addLayout(self.HBL1)

        self.HBL2 = QHBoxLayout()
        self.LabelScale2 = QLabel('Cam2 Scale (1px = ??cm) : ')
        self.EditLineScale2 = QLineEdit()
        self.HBL2.addWidget(self.LabelScale2)
        self.HBL2.addWidget(self.EditLineScale2)
        self.LayoutTotal.addLayout(self.HBL2)

        self.HBL3 = QHBoxLayout()
        self.LabelRes1 = QLabel('Cam1 Resolution : ')
        self.EditLineRes1 = QLineEdit()
        self.HBL3.addWidget(self.LabelRes1)
        self.HBL3.addWidget(self.EditLineRes1)
        self.LayoutTotal.addLayout(self.HBL3)
        
        self.HBL4 = QHBoxLayout()
        self.LabelRes2 = QLabel('Cam2 Resolution : ')
        self.EditLineRes2 = QLineEdit()
        self.HBL4.addWidget(self.LabelRes2)
        self.HBL4.addWidget(self.EditLineRes2)
        self.LayoutTotal.addLayout(self.HBL4)

        self.MAJ_BTN = QPushButton("Change Settings")
        self.MAJ_BTN.clicked.connect(self.ChangeSettings)
        self.LayoutTotal.addWidget(self.MAJ_BTN)

        # default params:
        self.default_scale1 = 1
        self.default_scale2 = 2 
        self.default_res1 = 1000
        self.default_res2 = 2000

        self.SetDefaultValues()
        self.UpdateValues()

        self.setLayout(self.LayoutTotal)
    
    def SetDefaultValues(self):
        self.scale1 = self.default_scale1
        self.scale2 = self.default_scale2
        self.res1 = self.default_res1
        self.res2 = self.default_res2
        

    
    def UpdateValues(self):
        self.EditLineScale1.setText(str(self.scale1))
        self.EditLineScale2.setText(str(self.scale2))
        self.EditLineRes1.setText(str(self.res1))
        self.EditLineRes2.setText(str(self.res2))

    def ChangeSettings(self):
        try:
            self.scale1 = int(self.EditLineScale1.text())
            self.scale2 = int(self.EditLineScale2.text())
            self.res1 = int(self.EditLineRes1.text())
            self.res2 = int(self.EditLineRes2.text())
            print("Values updates successfully !")
        except:
            self.SetDefaultValues()
            print("Invalid entered Value(s) ...")
            print("Setting back default values")
        
        
        self.UpdateValues()
        print(self.scale1, self.scale2, self.res1, self.res2)
        
        
        
        
if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = CamSettings_Widget()
    Root.show()
    sys.exit(App.exec())  

