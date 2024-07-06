import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import cv2

class ArduinoSettings_Widget(QWidget):
    def __init__(self):
        super(ArduinoSettings_Widget, self).__init__()
        #self.resize(1000, 1000)

        self.setWindowTitle("Arduino Control Panel - GyroSpin App")

        #---   Layout total ---#
        self.LayoutTotal = QVBoxLayout()


        self.FreqHBLyout = QHBoxLayout()
        self.FreqLabel = QLabel('Drive Frequency (Hz) :')
        self.FreqLineEdit = QLineEdit()
        #self.FreqEditButton = QPushButton('Edit Drive Frequency')
        self.FreqHBLyout.addWidget(self.FreqLabel)
        self.FreqHBLyout.addWidget(self.FreqLineEdit)
        #self.FreqHBLyout.addWidget(self.FreqEditButton)
        # No Need

        self.LayoutTotal.addLayout(self.FreqHBLyout)

        self.EmptySpace1 = QLabel()
        self.LayoutTotal.addWidget(self.EmptySpace1)

        self.StartDriveBTN = QPushButton('Start Drive')
        self.StartDriveBTN.clicked.connect(self.StartDrive)
        self.LayoutTotal.addWidget(self.StartDriveBTN)
    
        self.EmptySpace2 = QLabel()
        self.LayoutTotal.addWidget(self.EmptySpace2)

        self.PhaseHBLayout = QHBoxLayout()
        self.PhaseLabel = QLabel('Estimated Phase (degrees) :')
        self.PhaseValue = QLabel()
        self.PhaseHBLayout.addWidget(self.PhaseLabel)
        self.PhaseHBLayout.addWidget(self.PhaseValue)

        self.LayoutTotal.addLayout(self.PhaseHBLayout)

        self.setLayout(self.LayoutTotal)

        self.Freq = None
        self.Phase = None
        self.FreqLineEdit.setText("None")
        self.PhaseValue.setText("None")
        

    def ComputePhase(self):
        '''Calcul de la phase.'''
        return None

    def StartDrive(self):
        '''Lance le forçage.'''
        self.ComputePhase()
        try:
            self.Freq = float(self.FreqLineEdit.text())
            print("Starting Drive")
        except:
            self.Freq = None
            print("Invalid value of Freq")
        print(self.Freq)



if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = ArduinoSettings_Widget()
    Root.show()
    sys.exit(App.exec())    
        