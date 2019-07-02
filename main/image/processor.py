import numpy as np
import cv2 as cv

IMAGE_STOCK = '../../stock/receipt/'

class ReceiptImage:
    def __init__(self,filename):
        self.url = IMAGE_STOCK+filename
        self.c = 2
        self.blocksize = 11

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


ri = ReceiptImage('46.png')
img = ri.readImage(image_type = cv.IMREAD_GRAYSCALE)
ri.showImage(img)
c = 0
for x in range(2,20):
    print(x)
    th1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,x)
    while True:
        cv.imshow(str(x),th1)
        key = cv.waitKey(10) & 0xFF

        if key == ord('q'):
            # Press key `q` to quit the program
            cv.destroyAllWindows()
            exit() 

        elif key == ord('n'):
            cv.destroyAllWindows()
            break

        elif key == ord('s'):
            c = x
            cv.destroyAllWindows()
            break
    
    if c>0:
        break

print("blocksize")
print(c)

for i in range(3,40):
    if i%2 != 0: 
        print(i)
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,i,c)

        while True:
            cv.imshow(str(i),th2)
            key = cv.waitKey(10) & 0xFF

            if key == ord('q'):
                # Press key `q` to quit the program
                cv.destroyAllWindows()
                exit() 

            elif key == ord('n'):
                cv.destroyAllWindows()
                break