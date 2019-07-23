import numpy as np
import cv2 as cv
import imutils
import os
import yaml
from image.transform import *
from image.crop import ImageCroper
from utils.find_real_path import *
IMAGE_STOCK = '../../stock/receipt/'

class ReceiptImage:
    def __init__(self,url,filename,TRANSACTION_NUM):
        self.url = url
        self.c = 2
        self.blocksize = 11

        self.dilated = None
        self.drawed = None
        self.warped = None 

        self.status = False
        self.filename = filename
        self.image = self.readImage()
        self.orig = self.image.copy()
        path = os.path.dirname(url)
        print(path)

        self.save_folder = '/image_processing/'
        if not os.path.exists(path+self.save_folder):
            os.makedirs(path+self.save_folder)
        # Save url
        self.dilated_url = path+self.save_folder+'dilated_'+filename
        self.drawed_url = path+self.save_folder+'drawed_'+filename
        self.wraped_url = path +self.save_folder+ 'wraped_'+filename

        with open('config.yml', 'rb') as f:
            self.conf = yaml.load(f.read())        
        

    def showImage(self, image=None, image_type = cv.IMREAD_UNCHANGED):
        """[Show image with custom configuration]
        
        Keyword Arguments:
            image {[type]} -- [description] (default: {None})
            image_type {[type]} -- [description] (default: {cv.IMREAD_UNCHANGED})
        """
        if image is None:
            img = cv.imread(self.url,image_type)
            cv.imshow('default image',img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            cv.imshow('custom image',image)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def readImage(self, image_type = cv.IMREAD_UNCHANGED):
        """[Read image with custom configuration]
        
        Keyword Arguments:
            image_type {[type]} -- [description] (default: {cv.IMREAD_UNCHANGED})
        
        Returns:
            [type] -- [description]
        """
        return cv.imread(self.url, image_type)

    def find_Threshold(self,image, type = cv.ADAPTIVE_THRESH_GAUSSIAN_C, c_range = range(2,20)):
        """[Find suitable C in the argument]
        
        Usage:
            Press Q for quit

            Press C for increase C
            Press B for increase Block size

            Press S for save the var

        Arguments:
            image {[type]} -- [description]
        
        Keyword Arguments:
            type {[type]} -- [description] (default: {cv.ADAPTIVE_THRESH_GAUSSIAN_C})
            c_range {[type]} -- [description] (default: {range(2,20)})
        """
        while True:
            done = False
            th1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,self.blocksize,i)

            while True:
                cv.imshow(str(i),th1)
                key = cv.waitKey(10) & 0xFF

                if key == ord('q'):
                    # Press key `q` to quit the program
                    cv.destroyAllWindows()
                    exit() 

                elif key == ord('c'):
                    # Press 
                    self.c += 1



                elif key == ord('s'):
                    done = True
                    cv.destroyAllWindows()
                    break
            
            if done:
                break

    def processImage(self):
        """[Process Image]
        """
        # print(self.url)
        import os
        print("PROCESS IMAGE")
        print(self.url)
        print(os.path.isfile(self.url))
        self.image = cv.imread(self.url, 0)
        ic = ImageCroper(self.image)
        rect = ic.findPaper()

        self.dilated = ic.dilated
        self.drawed = ic.drawed

        # Transform
        if rect is None:
            self.status = False
            rect = self.getFullImageConner()
        else:
            self.status = True

        if self.status:
            self.warped = four_point_transform(self.orig, rect.reshape(4, 2))
        else:
            self.warped = self.drawed.copy()
        self.saveImage()

    def saveImage(self):
        

        cv.imwrite(self.dilated_url, self.dilated)
        cv.imwrite(self.drawed_url, self.drawed)
        cv.imwrite(self.wraped_url, self.warped)


    def getFullImageConner(self):
        return np.array([
            [0, 0], 
            [0,self.image.shape[0]],
            [self.image.shape[1], self.image.shape[0]],
            [self.image.shape[1],0]            
            ]
            )

    def rectCalibrate(self,nparray):
        height, width, channels = self.orig.shape 
        
        CALIBRATE = self.conf['CALIBRATION']['RATE']
        print("OLD %s",nparray)
        # result = []
        if self.conf['CALIBRATION']['ENABLE']:
            for i in range(0,len(nparray)):
                if i%2 == 0 :
                    cont = width
                else:
                    cont = height

                if (int(i/2)) == 0:
                    newvalue = nparray[i] - int(cont*CALIBRATE)
                elif (int(i/2)) == 1:
                    if i%2==0:
                        newvalue = nparray[i] + int(cont*CALIBRATE)
                    else:
                        newvalue = nparray[i] - int(cont*CALIBRATE)
                elif (int(i/2)) == 3:
                    if i%2==0:
                        newvalue = nparray[i] - int(cont*CALIBRATE)
                    else:
                        newvalue = nparray[i] + int(cont*CALIBRATE)
                else:
                    newvalue = nparray[i] + int(cont*CALIBRATE)

                if (i%2) == 0:
                    # x
                    if newvalue <= 0 or newvalue >= width:
                        continue
                else:
                    # y
                    if newvalue <= 0 or newvalue >= height:
                        continue
                nparray[i] = newvalue
            # for j in range(0,len(nparray[i])):
            #     print(nparray[i][j])
        print("NEW %s",nparray)
        return nparray

    def getNumPyfromConner(self,array):
        return np.array([
            [array[0], array[1]], 
            [array[2],array[3]],
            [array[4], array[5]],
            [array[6],array[7]]            
            ]
            )

    def transformImage(self,text_file):
        result = []
        path = os.path.dirname(self.url)
        file = open(text_file, "r") 
        i = 0
        save_folder = self.save_folder + 'croped/'

        if not os.path.exists(path+save_folder):
            os.makedirs(path+save_folder)
        print("TRANSFORM IMAGE")
        print(path+save_folder)
        for line in file: 
            filename = path+save_folder+str(i)+'_'+self.filename
            array = line.split(',')[:-1]
            array = list(map(int, array))

            filetest = path+save_folder+'tested'+str(i)+'_'+self.filename
            imagetest = self.orig.copy()

            pts = np.array(array[:8], np.int32)
            pts = pts.reshape((-1,1,2))

            cv2.polylines(imagetest, [pts], True, color=(0, 255, 0),
                                thickness=2)
            cv.imwrite(filetest, imagetest)
            replace_filename= get_url_path(filetest)
            result.append(replace_filename)


            # array = self.rectCalibrate(array)
            rect = self.getNumPyfromConner(array)
            image = four_point_transform(self.orig, rect.reshape(4, 2))
            cv.imwrite(filename, image)
            replace_filename= get_url_path(filename)
            result.append(replace_filename)

            i+=1

        return result

    def transformImageByBoxes(self,boxes):
        result = []
        path = os.path.dirname(self.url)

        for i, box in enumerate(boxes):
            filename = path+'/'+'croped_'+str(i)+'_'+self.filename

            filetest = path+'/'+'croped_tested'+str(i)+'_'+self.filename
            imagetest = self.orig.copy()
            cv2.polylines(imagetest, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                            thickness=2)
            # imagetest = cv2.resize(imagetest, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            cv.imwrite(filetest, imagetest)
            replace_filename= get_url_path(filetest)
            result.append(replace_filename)


            image = four_point_transform(self.orig, box[:8].reshape(4, 2))
            cv.imwrite(filename, image)
            replace_filename= get_url_path(filename)
            result.append(replace_filename)
        
        return result

        # saved_path = 
    

# ri = ReceiptImage('1.jpeg')
# ri.processImage()

# img = ri.readImage(image_type = cv.IMREAD_GRAYSCALE)
# ri.showImage(img)

# ic = ImageCroper(img)
# ic.findPaper()

# ri.showImage(ic.drawed)
# gray = cv.GaussianBlur(img, (5, 5), 0)

# kernel = np.ones((3, 3), 'uint8')

# # parameters
# window_scale = 0.8
# k_size = 3
# erode_iterations = 2
# # convert the target image to gray-scale
# target_image_gray = img

# # create a window for adaptive_thresholding that fits inside the image (nb: needs to be odd number)
# window_raw = int(min(target_image_gray.shape) * window_scale)
# adaptive_window = int(window_raw if window_raw % 2 == 1 else window_raw - 1)

# # run adaptive thresholding
# thresh_adapt = cv.adaptiveThreshold(target_image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
#                                         adaptive_window, 1)

# # quick filter - erosion to remove 'white' noise outside the A4
# kernel = np.ones((k_size, k_size), 'uint8')
# cleaned_area = cv.erode(thresh_adapt, kernel, iterations=erode_iterations)

# ri.showImage(cleaned_area)
# img = cleaned_area
# can = cv.Canny(img, 75, 200)
# ri.showImage(can)
# cnts = cv.findContours(can.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]

# # loop over the contours
# for c in cnts:
#     # approximate the contour
#     peri = cv.arcLength(c, True)
#     approx = cv.approxPolyDP(c, 0.02 * peri, True)

#     t = cv.drawContours(img.copy(),c, -1, (0, 255, 0), 2)
#     cv.imshow("utp",t)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     # if our approximated contour has four points, then we
#     # can assume that we have found our screen
#     if len(approx) == 4:
#         screenCnt = approx
#         break

# # show the contour (outline) of the piece of paper
# print("STEP 2: Find contours of paper")
# cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
# cv.imshow("Outline", image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
# #                 cv.THRESH_BINARY,25,8)
# # ri.showImage(th)

# c = 0

# for x in range(1,20):
#     x*=10
#     print(x)
#     gray = cv.GaussianBlur(img, (5, 5), 0)
#     th1 = cv.Canny(gray, 75, x)
#     while True:
#         cv.imshow(str(x),th1)
#         key = cv.waitKey(10) & 0xFF

#         if key == ord('q'):
#             # Press key `q` to quit the program
#             cv.destroyAllWindows()
#             exit() 

#         elif key == ord('n'):
#             cv.destroyAllWindows()
#             break

#         elif key == ord('s'):
#             c=x
#             cv.destroyAllWindows()
#             break
    
#     if c>0:
#         break
# c = 0
# for x in range(2,20):
#     print(x)
#     th1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                 cv.THRESH_BINARY,11,x)
#     while True:
#         cv.imshow(str(x),th1)
#         key = cv.waitKey(10) & 0xFF

#         if key == ord('q'):
#             # Press key `q` to quit the program
#             cv.destroyAllWindows()
#             exit() 

#         elif key == ord('n'):
#             cv.destroyAllWindows()
#             break

#         elif key == ord('s'):
#             c = x
#             cv.destroyAllWindows()
#             break
    
#     if c>0:
#         break

# print("blocksize")
# print(c)

# for i in range(3,40):
#     if i%2 != 0: 
#         print(i)
#         th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                     cv.THRESH_BINARY,i,c)

#         while True:
#             cv.imshow(str(i),th2)
#             key = cv.waitKey(10) & 0xFF

#             if key == ord('q'):
#                 # Press key `q` to quit the program
#                 cv.destroyAllWindows()
#                 exit() 

#             elif key == ord('n'):
#                 cv.destroyAllWindows()
#                 break