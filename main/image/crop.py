import cv2
import numpy as np 
import transform
import yaml

class ImageCroper:
    def __init__(self,image):
        # load parameters
        with open('../config.yml', 'rb') as f:
            self.conf = yaml.load(f.read())

        # load image
        self.orig = image.copy()
        self.image = image
        self.drawed = image.copy()
        self.dilated = image.copy()

    def findPaper(self):
        """[Find the receipt inside the image]
        """
        # At first, we have to load the param from the configuration file
        # But wait, oh it's loaded
        # The first param is blur param
        blur_param = self.conf['IMAGE_CROPER']['BLUR']
        # Canny param
        canny_param = self.conf['IMAGE_CROPER']['CANNY']
        # Hough param
        hough_param = self.conf['IMAGE_CROPER']['HOUGH']
        # Morph param
        morph_param = self.conf['IMAGE_CROPER']['MORPH']

        # Convert image to grayscale

        if len(self.image.shape)== 3:
            self.image =  cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Give a little blur
        cv2.GaussianBlur(self.image, (blur_param,blur_param), 0, self.image)

        # this is to recognize white on white
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(morph_param,morph_param))
        self.dilated = cv2.dilate(self.image, kernel)

        # Find edge
        edges = cv2.Canny(self.dilated, 0, canny_param, apertureSize=3)

        
        lines = cv2.HoughLinesP(edges, 1,  3.14/180, hough_param)
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
            rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
            rects.append(rect)

        final_rect = self.getLargestRect(rects)

        # draw the contour 
        cv2.drawContours(self.drawed, rects,-1,(0,255,0),1)

        return final_rect
    

    def getLargestRect(self,rects):
        largest_cnt = None
        area = 0
        if rects:
            for rect in rects:
                current_area = cv2.contourArea(rect)

                if area == 0:
                    # Asign the first into area
                    area = current_area
                    largest_cnt = rect
                elif area == current_area:
                    # if have the same area
                    # find on number of verties
                    if len(largest_cnt) < len(rect):
                        area = current_area
                        largest_cnt = rect
                elif area < current_area:
                    # If have larger area
                    # Found it
                    area = current_area
                    largest_cnt = rect

        return largest_cnt


# a = ImageCroper("f")