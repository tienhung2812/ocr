import cv2, numpy as np
import sys

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new,new)
    return new

if __name__ == '__main__':
    orig = cv2.imread(sys.argv[1])

    # these constants are carefully picked
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3,3), 0, img)

    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.dilate(img, kernel)

    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    for line in lines[0]:
         cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                         (255,0,0), 2, 8)

    # finding contours
    image, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        epsilon = 0.1*cv2.arcLength(cont,True)
        rect = cv2.approxPolyDP(cont, epsilon, True).copy().reshape(-1, 2)
        rects.append(rect)


    # that's basically it
    cv2.drawContours(orig, rects,-1,(0,255,0),1)

    # show only contours
    new = get_new(img)
    cv2.drawContours(new, rects,-1,(0,255,0),1)
    cv2.GaussianBlur(new, (9,9), 0, new)
    new = cv2.Canny(new, 0, CANNY, apertureSize=3)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('orig', orig)
    cv2.waitKey(0)
    # cv2.imshow('clean', cleaned_area)
    # cv2.waitKey(0)
    cv2.imshow('dilated', dilated)
    cv2.waitKey(0)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.imshow('new', new)
    cv2.waitKey(0)
