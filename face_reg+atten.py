import face_recognition
import cv2
import os
from PIL import Image
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

####################################First test pour apprendre ###########################################
'''imgElon = face_recognition.load_image_file('Images/6.png')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/5.png')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2) # top, right, bottom, left

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)


cv2.imshow('llll',imgElon)
cv2.imshow('llll1',imgTest)
cv2.waitKey(0) '''

labels = []
images = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #which return the path of the current directory
image_dir = os.path.join(BASE_DIR, "images/images")  # pour determiner le path vers les images en faire la concatenation de path et le nom de dossier qui contient les images



for root, dirs, files in os.walk(image_dir):
    for file in files :
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","_").upper()  # dirname : pour supprimer l'estension .png or .jpg // basename = pour obtenir seulement la derniere partie apres \
            pil_image = Image.open(path)
            #pil_image = cv2.imread(f'{path}/{file}')
            np_image = np.array(pil_image, "uint8")
            images.append(np_image)
            labels.append(label)

print(images)
print(labels)

############################ to load just the images in the same directory##################
'''path = 'Images/firas'
images = []     # LIST CONTAINING ALL THE IMAGES
className = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])
        #print(images)'''


def findEncodings(images):
    i=0
    
    encodeList = []
    for img in images:
        labels[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels[i]
        encode = face_recognition.face_encodings(img)[0]
        i+=1
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

print('Encodings Complete')
print(encodeListKnown)


with open("Encoding.pickle",'wb') as f:
    pickle.dump(encodeListKnown,f)
with open("Labels_Encoding.pickle",'wb') as f:
    pickle.dump(labels,f)

print('files saved')

#cap = cv2.VideoCapture('http://admin:123456@192.168.1.178/videostream.cgi?rate=0')
'''cap = cv2.VideoCapture(0)
while True:
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
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break'''


