import numpy as np
import cv2
from urllib.request import urlopen

from ssl import SSLContext,PROTOCOL_TLSv1

#cap = cv2.VideoCapture('http://admin:123456@192.168.1.178/videostream.cgi?rate=0')
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.18:554/1/h264major')


# admin:123456@      H264?ch=1&subtype=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows