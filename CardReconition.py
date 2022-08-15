import cv2
import numpy as np


"""
this function help the algorithm 
to see in a straight perspective
regardless of the angle of the card
"""

def card_prespective(cnt,img):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        p1 = approx[0][0]
        p2 = approx[1][0]
        p3 = approx[2][0]
        p4 = approx[3][0]

        width, height = 130, 130
        cardPoints = np.float32([[0, 0], [height, 0], [height, width], [0, width]])
        if np.linalg.norm(p1 - p2) > np.linalg.norm(p2 - p3):
            oldPoints = np.float32([p2, p1, p4, p3])
            matrix = cv2.getPerspectiveTransform(oldPoints, cardPoints)
            imgOut = cv2.warpPerspective(img, matrix, (width, height))

        else:
            oldPoints = np.float32([p1, p4, p3, p2])
            matrix = cv2.getPerspectiveTransform(oldPoints, cardPoints)
            imgOut = cv2.warpPerspective(img, matrix, (width, height))
        return imgOut



green = np.array([99,144,50])
red = np.array([8,24,194])
purple = np.array([117,74,72])

"""
Determines the color of the card 
by comparing it to the
average of the color samples
"""
def get_color(rgbAvg):
    min = float('inf')
    color = 0
    greenDist = np.linalg.norm(rgbAvg - green)
    redDist = np.linalg.norm(rgbAvg - red)
    purpleDist = np.linalg.norm(rgbAvg - purple)

    if greenDist < min:
        min = greenDist
        color = 0
    if redDist < min or rgbAvg[2] > 125:
        min = redDist
        color = 1
    if purpleDist < min and rgbAvg[2] < 135:
        min = purpleDist
        color = 2


    return color


"""
Determines the shade of the card 
according to the proportional size of the shape
in relation to the colored pixels in it
"""

def get_shade(thresh, img,countres):
    max_x, min_x = float('-inf'), float('inf')
    max_y, min_y = float('-inf'), float('inf')

    for c in countres:
        if c.shape[0] > 47:
            cont = c

            max_x, min_x = max([i[0][0] for i in cont]), min([i[0][0] for i in cont])
            max_y, min_y = max([i[0][1] for i in cont]), min([i[0][1] for i in cont])
            img = img[min_y:max_y, min_x:max_x]

            break
    try:
        thresh = thresh[min_y:max_y, min_x:max_x]
    except:
        print(min_y, " ", max_y, " ", min_x, " ", max_x)

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
    counter = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = np.sqrt(((x2 - x1) ** 2)) + ((y2 - y1) ** 2)
        if distance > 100:
            counter += 1

    if counter > 17:
        return 1
    return 2

"""
Determines the number of shapes in card 
according to the max largest contours
"""


def get_number(contours):
    edges = 0
    maxSize = 0
    for i in contours:
        maxSize = np.maximum(maxSize, int(i.shape[0]))
        if i.shape[0] > 40:
            edges += 1
    return edges,maxSize

elipseSize,elipseType = 58,1
waveSize,waveType = 87,2
rhombusSize ,rhombusType= 118,3


"""
Determines the type of the shapes in the card 
by comparing it to the
average of the shapes size samples
"""
def get_shape(shapeSize):
    waveDist = np.abs(shapeSize - waveSize)
    elipseDist = np.abs(shapeSize - elipseSize)
    rhombusDist = np.abs(shapeSize - rhombusSize)
    minDist = min(waveDist, elipseDist, rhombusDist)
    if minDist == waveDist:
        return 2
    elif minDist == elipseDist:
        return 0
    return 1