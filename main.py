import numpy as np
import cv2 as cv


def find_corners(contures):
    max_x, min_x = max([i[0][0] for i in x]), min([i[0][0] for i in x])
    max_y, min_y = max([i[0][1] for i in x]), min([i[0][1] for i in x])






def start(name):
  img = cv.imread('set1.jpg')
  imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  thresh = cv.threshold(imgray, 0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
  contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  # cv.drawContours(img,contours[0],-1,(0,255,0),3)
  for i in range(len(contours)):
      print('card number: ',i)
      x = contours[i]
      # 0,1,8 green 2,3,5,9 red 4,7,10,11 purple
      max_x,min_x = max([i[0][0] for i in x]),min([i[0][0] for i in x])
      max_y,min_y = max([i[0][1] for i in x]),min([i[0][1] for i in x])


      find_parameters(img[min_y:max_y, min_x:max_x])




def draw_edges(img):

    i = img.shape[0]
    j = img.shape[1]

    for row in range(i):
        for col in range(j):
            if img[row][col] == 0:
                break
            else:
                img[row][col] = 0

    for row in range(i):
        for col in range(j-1,0,-1):
            if img[row][col] == 0:
                break
            else:
                img[row][col] = 0

    for row in range(1,img.shape[0]-1):
        for col in range(1,img.shape[1]-1):
            if img[row][col] == 255:
                if img[row+1][col]==0 and img[row-1][col]==0 and img[row][col+1]==0 and img[row][col-1]==0:
                    img[row][col] = 0









    return img

def find_color(originalCard, BnWCard):
    i = originalCard.shape[0]
    j = originalCard.shape[1]
    counter = 0
    rgb = np.zeros(3)
    for row in range(i):
        for col in range(j):
            if BnWCard[row][col] == 255:
                counter+=1
                # print(originalCard[row][col])
                rgb += originalCard[row][col]



    print('rgb avg = ',rgb/counter)



def find_number_of_shapes(imgray):
    thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    approx = cv.approxPolyDP(contours[0], 0.01 * cv.arcLength(contours[0], True), True)
    return len(contours)

def find_shape():
    return 0

def find_pattern():
    return 0

def find_parameters(card):

    imgray = cv.cvtColor(card, cv.COLOR_BGR2GRAY)
    tempImg = np.zeros(imgray.shape)
    tempImg[imgray>150] = 0
    tempImg[imgray<150] = 255
    tempImg = draw_edges(tempImg)

    for row in range(imgray.shape[0]):
        for col in range(imgray.shape[1]):
            imgray[row][col] = tempImg[row][col]



    cv.imshow("imgray", imgray)
    cv.waitKey(0)

    numOfShapes = find_number_of_shapes(imgray)
    print('num of shapes in card is: ',numOfShapes)

    color = find_color(card, imgray)
    print('shapes color is: ', color)

    shape = find_shape()
    print('shapes is: ', shape)

    pattern = find_pattern()
    print('shape pattern is: ', pattern)




    # cv.drawContours(card,contours,-1,(0,0,255),3)
    # cv.imshow("card",card)
    # cv.waitKey(0)
    return numOfShapes,color,shape,pattern





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = [[2,2,2,2],[4,4,4,4]]
    x = np.array(x)
    # print(x)
    print(x/2)



    start('PyCharm')


