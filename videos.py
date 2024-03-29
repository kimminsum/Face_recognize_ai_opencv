### recognizing face and eyes ###
import numpy as np
import cv2
"""
"""
class Recognize:
    def __init__(self):
        ### Install part of body
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.main()
    """
    Main
    """
    def main(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            _, frame = video_capture.read()
            # frame = cv2.flip(frame,-1) ### reverse the image up to down direction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            canvas = self.detect(gray,frame)
            
            cv2.imshow('canvas',canvas)

            if cv2.waitKey(30) == 27: ### esc
                break
        video_capture.release()
        cv2.destroyAllWindows()
    """
    Detect Method
    """
    def detect(self, gray, frame):
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100),flags = cv2.CASCADE_SCALE_IMAGE)
        ### find face first and recognize eyes
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2) ### show face by rectangle
            
            face_gray=gray[y:y+h,x:x+w]
            face_color=frame[y:y+h,x:x+w]
            
            eyes = self.eyeCascade.detectMultiScale(face_gray,1.1,3)
            ### find and recognize eyes
            for(ex, ey,ew,eh) in eyes:
                cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) ### show eyes by rectangle
                
        return frame

if __name__=="__main__":
    Recognize()