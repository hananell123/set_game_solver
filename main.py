import numpy as np
import cv2
import CardReconition as cr
from numpy import random as rnd
import playsound
from card import Card
from SetFinder import find_sets
from threading import *
import time

originalImage = []
cap = []
interupt = False


def make_constrast(img):
    kernel = np.ones((5,5), np.float32) / 24
    img = cv2.filter2D(img, -1, kernel)


    return img


def get_card_center(width = 130,height = 130):
    x = int((width * 2) / 4)
    y = int((height * 2) / 4)
    return (x, y)


def start():
  global interupt

  global cap
  cap = cv2.VideoCapture(0)


  while (True):
      interupt = False
      allCards = []
      ret, cleanImg = cap.read()
      cv2.imshow("test", cleanImg)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      time.sleep(1.5)
      img = cleanImg.copy()
      global originalImage
      originalImage = img.copy()
      originalImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)

      alpha = 1.4
      beta = 20

      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      imgray = cv2.addWeighted(imgray, alpha, np.zeros(imgray.shape, imgray.dtype), 0, beta)
      thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


      for cnt in (contours):
          imgOut = cr.card_prespective(cnt,img)
          imgray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
          number,shape,color,shade,cont = get_card_parameters(imgray, imgOut)
          newCard = Card(shape,shade,color,number,cnt)

          cv2.drawContours(cleanImg, cnt, -1,(255, 0, 0), 4)
          time.sleep(0.5)
          allCards.append(newCard)

      numOfCards = len(allCards)
      T = Thread(target=interrupt_check)
      T.start()
      alreadyFound = False
      allSets = []
      while interupt == False:
          alreadyFound = display_set(numOfCards, allCards, cap, alreadyFound, allSets)
      after_iterrupt()



def after_iterrupt():
    global cap
    global interupt
    videoSamples = [100,100,100,100]
    _, im = cap.read()


    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    time.sleep(0.5)
    while True:


        _, im2 = cap.read()

        im3 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        diff_frame = cv2.absdiff(src1=im, src2=im3)
        im = im3

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        size = len(contours)
        videoSamples[3]=videoSamples[2]
        videoSamples[2] = videoSamples[1]
        videoSamples[1] = videoSamples[0]
        videoSamples[0] = size
        flag = True
        for sample in videoSamples:
            if sample>10:
                flag = False
                break
        if flag:
            interupt = True
            cv2.putText(im2, "V", (int(im2.shape[0] / 2), int(im2.shape[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 4)
            cv2.imshow("test", im2)
            cv2.waitKey(300)
            time.sleep(1)
            break
        time.sleep(0.1)
        cv2.putText(im2,"INTERUPT",(00,int(im2.shape[1]/2)),cv2.FONT_HERSHEY_SIMPLEX,3,(0, 0, 255),4)
        cv2.imshow("test", im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def interrupt_check():
    global cap
    global originalImage
    while True:
        ret, im = cap.read()
        im =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        diff_frame = cv2.absdiff(src1=originalImage, src2=im)
        originalImage = im

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        size = len(contours)
        if size > 70:
            global interupt
            interupt = True
            break
        time.sleep(0.3)


def display_set(numOfCards, allCards, cap, alreadyFound, allSets):
        if alreadyFound==False:

            for i in range(numOfCards):
                for j in range(i + 1, numOfCards):
                    for t in range(j + 1, numOfCards):
                        card1, card2, card3 = allCards[i], allCards[j], allCards[t]
                        _, tempIMG = cap.read()
                        cv2.drawContours(tempIMG, (card1.get_countres(), card2.get_countres(), card3.get_countres()), -1,
                                         (0, 0, 255), 4)
                        cv2.imshow("test", tempIMG)
                        cv2.waitKey(5)
                        if cv2.waitKey(1) == ord('q'):
                            break

                        if find_sets(card1, card2, card3):

                            allSets.append((card1.get_countres(), card2.get_countres(), card3.get_countres()))
                            cv2.drawContours(tempIMG, (card1.get_countres(), card2.get_countres(), card3.get_countres()), -1,
                                             (0, 255, 0), 4)
                            cv2.imshow("test", tempIMG)
                            cv2.waitKey(750)
            return True
        else:
            for set in allSets:
                _, tempIMG = cap.read()
                cv2.drawContours(tempIMG, set, -1,
                                 (0, 255, 0), 4)
                cv2.imshow("test", tempIMG)
                cv2.waitKey(700)



def get_card_parameters(imgray, original):

    thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    tempImg = np.zeros(thresh.shape)
    tempImg[thresh > 150] = 0
    tempImg[thresh < 150] = 255

    rgb = np.array([0, 0, 0])
    counter = 0
    for row in range(imgray.shape[0]):
        for col in range(imgray.shape[1]):
            thresh[row][col] = tempImg[row][col]
            if thresh[row][col]==255:
                counter+=1
                rgb+=original[row][col]

    rgb = np.divide(rgb,counter)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = cr.get_color(rgbAvg=rgb)
    shade = cr.get_shade(thresh, original,contours)
    number,maxSize = cr.get_number(contours)
    shape = cr.get_shape(maxSize)
    return number,shape,color,shade,contours


if __name__ == '__main__':
    start('PyCharm')

