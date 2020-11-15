import numpy as np
import cv2
from urllib.request import urlopen
import face_recognition
import cv2
import os
from PIL import Image
import pickle
import numpy as np

with open("Encoding.pickle",'rb') as f:
    Encoding= pickle.load(f)
    print (Encoding)

with open("Labels_Encoding.pickle",'rb') as f:
    Labels_Encoding = pickle.load(f)
    print(Labels_Encoding)
print('both encoding')
encodeListKnown = Encoding
labels = Labels_Encoding


#cap = cv2.VideoCapture('http://admin:123456@192.168.1.178/videostream.cgi?rate=0')
#cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.18:554/1/h264major')
cap = cv2.VideoCapture(0)
#cap.set(cv.CAP_PROP_FPS,10)
#print(cap.get(cv2.CV_CAP_PROP_FPS))

target = 30 
counter = 0
# admin:123456@      H264?ch=1&subtype=0
while(True):
    # Capture frame-by-frame
    if counter == target: 
    
        # Display the resulting frame
       
        counter = 0 
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = labels[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            else:
                name = 'Unknown'
        # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('webcam',img)
    else: 
        success = cap.grab() 
        counter += 1     
    print(counter)

    # Our operations on the frame come here
   
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows




#cap = cv2.VideoCapture('http://admin:123456@192.168.1.178/videostream.cgi?rate=0')
