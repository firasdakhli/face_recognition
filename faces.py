import numpy as np
import cv2
import pickle
from datetime import datetime

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/trainner.yml")

labels = {}
with open("pickles_file/Label.pickle",'rb') as f:
    labels = pickle.load(f)
    labels = { k:v for v,k in labels.items()}
    print(labels)

def markAttendance(name):
    with open('Attendance.cvs','w+') as f:
        myDataList = f.readlines()
        nameList = []
       # for line in myDataList:
        #    entry = line.split(',')
         #   nameList.append(entry[0])
        #if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString}')


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:

        #print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        #matchIndex = np.argmin(conf)
        #print(matchIndex)
        if conf>=40 and conf<= 55:
            print(100 - conf)
            print(labels[id_])
            markAttendance(labels[id_])
            stroke = 2
            cv2.putText(frame, labels[id_] , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),stroke, cv2.LINE_AA)
            cv2.putText(frame," score: " + str(100-conf), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255))

        #to write or to save the image
        #img_item = "Mootez.png"
        #cv2.imwrite(img_item, roi_color)
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows