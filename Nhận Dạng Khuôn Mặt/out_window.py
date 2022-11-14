
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer,QDate,Qt
from PyQt5.QtWidgets import QDialog
import cv2
import face_recognition
import numpy as np
import datetime
import os


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        now = QDate.currentDate()
        concurrent_data = now.toString("dd/MM/yyyy")
        cr_time = datetime.datetime.now().strftime("%I:%M %p")
        self.lb_time.setText(cr_time)
        self.lb_ngay.setText(concurrent_data)
        self.image = None

    @pyqtSlot()
    def startVideo(self, camera_name):

        if len(camera_name) == 1:
        	self.capture = cv2.VideoCapture(int(camera_name))
        else:
        	self.capture = cv2.VideoCapture(camera_name)
        self.timer = QTimer(self)
        path = 'ImagesAttendance'
        if not os.path.exists(path):
            os.mkdir(path)

        images = []
        self.class_names = []
        self.encode_list = []
        attendance_list = os.listdir(path)

        for cl in attendance_list:
            cur_img = cv2.imread(f'{path}/{cl}')
            images.append(cur_img)
            self.class_names.append(os.path.splitext(cl)[0])
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img)
            encodes_cur_frame = face_recognition.face_encodings(img, boxes)[0]

            self.encode_list.append(encodes_cur_frame)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(40)
    def face_rec_(self, frame, encode_list_known, class_names):

        def mark_attendance(name):

            with open('FileInfo.txt', 'a') as f:
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                f.writelines(f'\n{name},{date_time_string}')
        # face recognition
        faces_cur_frame = face_recognition.face_locations(frame)
        encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)

        for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):

            matches = face_recognition.compare_faces(encode_list_known, encodeFace)
            faceDis = face_recognition.face_distance(encode_list_known, encodeFace)
            print(faceDis)

            matchIndex = np.argmin(faceDis)
            print(matchIndex)

            if faceDis[matchIndex] < 0.50:
                name = class_names[matchIndex].upper()

            else:
                name = 'Unknown'
            # print(name)
            y1, x2, y2, x1 = faceLoc

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 51), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (255, 255, 51), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),1)
            mark_attendance(name)
        return frame

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.displayImage(self.image, self.encode_list, self.class_names, 1)

    def displayImage(self, image, encode_list, class_names, window=1):

        image = cv2.resize(image, (640, 480))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
