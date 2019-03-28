#Setup for linux
import os

#SETTING
INSTALL_CMAKE = True
INSTALL_TESSERACT = True
INSTALL_VIE_PACK = True
INSTALL_PYTHON3_PACKAGE = True
INSTALL_IMAGEMAGICK = True
#Requirements pacakge
#Unzip

#Install CMake
if INSTALL_CMAKE:
    print('Install CMake ...')
    os.system('sudo apt install cmake')

#install Tesseract
if INSTALL_TESSERACT:
    print('Install Tesseract ...')
    os.system('sudo apt install tesseract-ocr=4.00~git2288-10f4998a-2')
    tesspath = '/usr/share/tesseract-ocr/4.00'

# Download Tesseract vietnamese package
if INSTALL_VIE_PACK:
    print('Install Tesseract Vietnamese package...')
    # viepackfolder = 'tesseract-ocr-3.02.vie'
    # viepack = viepackfolder+'.zip'
    # os.system('wget https://excellmedia.dl.sourceforge.net/project/vietocr/lang%20data%20for%20tesseract-ocr/vietnamese%20language%20pack/'+viepack)
    # os.system('unzip '+viepack)
    # os.system('sudo cp '+viepackfolder+'/tesseract-ocr/tessdata/vie.traineddata '+tesspath+'/tessdata')
    os.system('sudo apt install tesseract-ocr-vie')

#Instqll python3 package
if INSTALL_PYTHON3_PACKAGE:
    print('Install python3 package')
    os.system('pip3 install pillow')
    os.system('pip3 install pytesseract')
    os.system('pip3 install wand')
    # os.system('pip3 install pyslibtesseract')

if INSTALL_IMAGEMAGICK:
    print('Install ImageMagick')
    os.system('sudo apt-get install libmagickwand-dev')
    print('Config ImageMagick Policy')
    os.system('sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xmlout')