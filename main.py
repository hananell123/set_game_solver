import numpy as np
import cv2 as cv



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








def start(name):
  cap = cv.VideoCapture(0)

  while (True):
      ret, img = cap.read()
      imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
      contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

      for cnt in (contours):
          approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)

          p1 = approx[0][0]
          p2 = approx[1][0]
          p3 = approx[2][0]
          p4 = approx[3][0]


          cv.drawContours(img,approx,-1,(0,255,0),3)
          cv.circle(img,[1,120],3,(0,0,255),cv.FILLED)
          # cv.imshow('regular', img)
          # cv.waitKey(0)


          width,height = 130,130
          cardPoints = np.float32([[0, 0],[height, 0],[height, width],[0, width] ])
          if np.linalg.norm(p1-p2)>np.linalg.norm(p2 - p3):
              oldPoints = np.float32([p2,p1,p4,p3])
              matrix = cv.getPerspectiveTransform(oldPoints,cardPoints)
              imgOut = cv.warpPerspective(img,matrix,(width,height))

          else:
              oldPoints = np.float32([p1,p4,p3,p2])
              matrix = cv.getPerspectiveTransform(oldPoints,cardPoints)
              imgOut = cv.warpPerspective(img,matrix,(width,height))
          edges = cv.Canny(imgOut, 80, 80)



          imgray = cv.cvtColor(imgOut, cv.COLOR_BGR2GRAY)

          x = int((width * 2) / 4)
          y = int((height * 2) / 4)
          center = (x, y)
          num,type,color,fill = find_number_of_shapes(imgray,imgOut,center)


          if type == 1:
              print("Count:",num,', shape: elipse, color:',color)
          elif type ==2:
              print("Count:", num,', shape: wave, color:',color)
          else:
              print("Count:", num, ', shape: rhom, color:',color)
          print('fill of one', fill / num)
          cv.imshow('COLOR card', imgOut)
          cv.waitKey(0)


      cv.imshow('frame', img)
      if cv.waitKey(1) & 0xFF == ord('q'):
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
    if redDist < min:
        min = redDist
        color = 'red'
    if purpleDist < min :#or rgbAvg[1]<110
        min = purpleDist
        color = 'purple'

    return color

def find_fill(thresh,original):
    canny = cv.Canny(original, 80, 80)
    counter = 0
    blackCounter=0
    for row in range(thresh.shape[0]):
        for col in range(thresh.shape[1]):
            if thresh[row][col]==255:
                blackCounter+=1
                if canny[row][col]==0:
                    counter+= 1

    return (blackCounter - counter)/blackCounter




def find_number_of_shapes(imgray,original,center = None,):

    # cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    fill = find_fill(thresh,original)

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

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # approx = cv.approxPolyDP(contours[0], 0.01 * cv.arcLength(contours[0], True), True)
    edges = 0
    maxSize = 0
    for i in contours:
        maxSize = np.maximum(maxSize,int(i.shape[0]))
        if i.shape[0]>40:
            edges+=1


    type = find_shape(maxSize)
    if center!= None:
        cv.putText(imgray, str(edges), center, cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


    return edges,type,color,fill/edges

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
