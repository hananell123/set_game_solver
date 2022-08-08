import numpy as np
import cv2
from numpy import random as rnd


elipseSize,elipseType = 58,1
waveSize,waveType = 87,2
rhombusSize ,rhombusType= 118,3

green = np.array([99,144,50])
red = np.array([8,24,194])
purple = np.array([117,74,72])








def find_shape(shapeSize):
    waveDist = np.abs(shapeSize-waveSize)
    elipseDist = np.abs(shapeSize-elipseSize)
    rhombusDist = np.abs(shapeSize-rhombusSize)

    minDist = min(waveDist,elipseDist,rhombusDist)

    if minDist == waveDist:
        return 2
    elif minDist == elipseDist:
        return 1
    return 3





def make_constrast(img):
    kernel = np.ones((5,5), np.float32) / 24
    img = cv2.filter2D(img, -1, kernel)

    # cv2.imshow('frame', img)
    # cv2.waitKey(0)

    return img
def start(name):
  cap = cv2.VideoCapture(0)
  size = 12


  while (True):
      fillMap = {
          0: "full",
         1: "stripes",
          2: "empty",
          -1:"ERROR"
      }

      ret, img = cap.read()
      print(len(img))
      # cv2.imshow('frame', img)
      # cv2.waitKey(0)

      # img = make_constrast(img)


      alpha = 1.4
      beta = 20
      # img = cv2.addWeighted(img,alpha,np.zeros(img.shape,img.dtype),0,beta)

      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      imgray = cv2.addWeighted(imgray, alpha, np.zeros(imgray.shape, imgray.dtype), 0, beta)
      thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      for cnt in (contours):
          fill = -1

          approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

          p1 = approx[0][0]
          p2 = approx[1][0]
          p3 = approx[2][0]
          p4 = approx[3][0]


          # cv2.drawContours(img, approx, -1, (0, 255, 0), 3)
          # cv2.circle(img, [1, 120], 3, (0, 0, 255), cv2.FILLED)
          # cv.imshow('regular', img)
          # cv.waitKey(0)


          width,height = 130,130
          cardPoints = np.float32([[0, 0],[height, 0],[height, width],[0, width] ])
          if np.linalg.norm(p1-p2)>np.linalg.norm(p2 - p3):
              oldPoints = np.float32([p2,p1,p4,p3])
              matrix = cv2.getPerspectiveTransform(oldPoints, cardPoints)
              imgOut = cv2.warpPerspective(img, matrix, (width, height))

          else:
              oldPoints = np.float32([p1,p4,p3,p2])
              matrix = cv2.getPerspectiveTransform(oldPoints, cardPoints)
              imgOut = cv2.warpPerspective(img, matrix, (width, height))
          edges = cv2.Canny(imgOut, 80, 80)



          imgray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)

          x = int((width * 2) / 4)
          y = int((height * 2) / 4)
          center = (x, y)
          num,type,color,fill = find_number_of_shapes(imgray,imgOut,center)


          if type == 1:
              print("Count:",num,', shape: elipse, color:',color,', fill is:',fillMap[fill])
          elif type ==2:
              print("Count:", num,', shape: wave, color:',color,', fill is:',fillMap[fill])
          else:
              print("Count:", num, ', shape: rhom, color:',color,', fill is:',fillMap[fill])

          cv2.imshow('COLOR card', imgOut)
          cv2.waitKey(0)


      cv2.imshow('frame', img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break


def find_color(rgbAvg):
    min = float('inf')
    color = ''
    greenDist = np.linalg.norm(rgbAvg - green)
    redDist = np.linalg.norm(rgbAvg - red)
    purpleDist = np.linalg.norm(rgbAvg - purple)

    if greenDist < min:
        min = greenDist
        color = 'green'
    if redDist < min or rgbAvg[2]>125:
        min = redDist
        color = 'red'
    if purpleDist < min and rgbAvg[2]<150:#or rgbAvg[1]<110
        min = purpleDist
        color = 'purple'

    print('avrg',rgbAvg)
    print('green:',greenDist)
    print('red',redDist)
    print('purple',purpleDist)

    return color
def fill(img,countres,thresh):
    max_x, min_x = float('-inf'), float('inf')
    max_y, min_y = float('-inf'), float('inf')
    cont = []
    for c in countres:
        if c.shape[0] > 50:
            cont = c
            # tmax_x, tmin_x = max([i[0][0] for i in cont]), min([i[0][0] for i in cont])
            # tmax_y, tmin_y = max([i[0][1] for i in cont]), min([i[0][1] for i in cont])
            max_x, min_x = max([i[0][0] for i in cont]), min([i[0][0] for i in cont])
            max_y, min_y = max([i[0][1] for i in cont]), min([i[0][1] for i in cont])
            img = img[min_y:max_y, min_x:max_x]
            # canny = cv2.Canny(img, 60, 60)
            # newImg = cv2.resize(canny, (30, 95))
            break



    thresh = thresh[min_y:max_y, min_x:max_x]


    whitePixelCounter = 0
    imgSize = thresh.shape[0]*thresh.shape[1]
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i][j] == 255:
                whitePixelCounter+=1

    # 0 = full
    # 1 = stripes
    # 2 = empty
    if whitePixelCounter/imgSize >0.5:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 5, 20)

    lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 20, maxLineGap=4)
    # 2 less then 17
    counter = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = np.sqrt(((x2 - x1) ** 2)) + ((y2 - y1) ** 2)
        if distance > 100:
            counter += 1


    if counter>17:
        return 1
    return 2





def find_fill(thresh,img,countres):
    max_x, min_x = float('-inf'), float('inf')
    max_y, min_y = float('-inf'), float('inf')
    for c in countres:
        if c.shape[0] > 50:
            cont = c
            # tmax_x, tmin_x = max([i[0][0] for i in cont]), min([i[0][0] for i in cont])
            # tmax_y, tmin_y = max([i[0][1] for i in cont]), min([i[0][1] for i in cont])
            max_x, min_x = max([i[0][0] for i in cont]), min([i[0][0] for i in cont])
            max_y, min_y = max([i[0][1] for i in cont]), min([i[0][1] for i in cont])
            img = img[min_y:max_y, min_x:max_x]
            # canny = cv2.Canny(img, 60, 60)
            # newImg = cv2.resize(canny, (30, 95))
            break

    thresh = thresh[min_y:max_y, min_x:max_x]

    whitePixelCounter = 0
    imgSize = thresh.shape[0] * thresh.shape[1]
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i][j] == 255:
                whitePixelCounter += 1

    # 0 = full
    # 1 = stripes
    # 2 = empty
    if whitePixelCounter / imgSize > 0.5:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 5, 20)

    lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 20, maxLineGap=4)
    # 2 less then 17
    counter = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = np.sqrt(((x2 - x1) ** 2)) + ((y2 - y1) ** 2)
        if distance > 100:
            counter += 1

    if counter > 17:
        return 1
    return 2





def find_number_of_shapes(imgray,original,center = None,):



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
    color = find_color(rgbAvg=rgb)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = find_fill(thresh, original,contours)
    # approx = cv.approxPolyDP(contours[0], 0.01 * cv.arcLength(contours[0], True), True)
    edges = 0
    maxSize = 0
    for i in contours:
        maxSize = np.maximum(maxSize,int(i.shape[0]))
        if i.shape[0]>40:
            edges+=1


    type = find_shape(maxSize)
    if center!= None:
        cv2.putText(imgray, str(edges), center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


    return edges,type,color,fill

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    start('PyCharm')




#
#
# def find_parameters(card):
#
#     # for i in range(0,card.shape[0]):
#     #     for j in range(0,card.shape[1]):
#     #         card[i][j][0] = min(abs(card[i][j][0]*1.2),255)
#     #         card[i][j][1] = min(abs(card[i][j][1] * 1.2), 255)
#     #         card[i][j][2] = min(abs(card[i][j][2] * 1.2), 255)
#     # cv.imshow('before', card)
#     # cv.waitKey(0)
#     # kernel = np.ones((5, 5), np.float32) / 25
#     # card = cv.filter2D(card, -1, kernel)
#     imgray = cv.cvtColor(card, cv.COLOR_BGR2GRAY)
#
#     edges = cv.Canny(imgray, 200, 200)
#
#     cv.imshow('before',card)
#     cv.waitKey(0)
#     cv.imshow('before',edges)
#     cv.waitKey(0)
#
#     tempImg = np.zeros(imgray.shape)
#     tempImg[imgray>150] = 0
#     tempImg[imgray<150] = 255
#     tempImg = draw_edges(tempImg)
#
#
#     for row in range(imgray.shape[0]):
#         for col in range(imgray.shape[1]):
#             imgray[row][col] = tempImg[row][col]
#
#     cv.imshow('after', imgray)
#     cv.waitKey(0)
#
#     numOfShapes = find_number_of_shapes(imgray)
#     print('Quantity: ',numOfShapes)
#
#     color = find_color(card, imgray)
#     print('Color: ', color)
#
#     shape = find_shape(imgray)
#     print('Shape: ', shape)
#
#     pattern = find_pattern()
#     print('Pattern: ', pattern)
#
#     cv.imshow("imgray", imgray)
#     cv.waitKey(0)
#
#
#
#
#     # cv.drawContours(card,contours,-1,(0,0,255),3)
#     # cv.imshow("card",card)
#     # cv.waitKey(0)
#     return numOfShapes,color,shape,pattern