import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import training.config as cf

def collect():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    counter = 0

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((cf.imgSize, cf.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - cf.offset:y + h + cf.offset, x - cf.offset:x + w + cf.offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = cf.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, cf.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((cf.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = cf.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (cf.imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((cf.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite(f'{cf.folder}/Image_{time.time()}.jpg',imgWhite)
            counter += 1
        
        if (counter == 500):
            return 

def run():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((cf.imgSize, cf.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - cf.offset:y + h + cf.offset, x - cf.offset:x + w + cf.offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = cf.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, cf.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((cf.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                k = cf.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (cf.imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((cf.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.rectangle(imgOutput, (x - cf.offset, y - cf.offset-50),
                        (x - cf.offset+90, y - cf.offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, cf.labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-cf.offset, y-cf.offset),
                        (x + w+cf.offset, y + h+cf.offset), (255, 0, 255), 4)
        
        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        if (key == ord("q")):
            return