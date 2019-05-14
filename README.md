# OCR
Vietnamese ocr for convert Scanned PDF to data

# Online Server
https://hung-ocr.herokuapp.com

## Installation
You can run ```python3 setup-linux.py``` or follow instruction below:

Package using : [tesseract-ocr](https://github.com/tesseract-ocr/tesseract)

- Install tesseract on linux 18
```
sudo apt install tesseract-ocr
```

- Install Vietnamese languagues data pack from [VietUnicode](http://vietunicode.sourceforge.net/howto/tesseract-ocr.html)

```
sudo apt-get install tesseract-ocr-vie
```
- Install ImageMagick

- Install Python requirements
```
pip3 install -r requirements.txt
```

## Run with docker
```
docker-compose up
```

## **Step 1:** Convert Scanned PDF to xml
[Folder](main/converter)  
[Document](main/converter/README.md)  
Usage:
`-m` : Method
`-i` : Image Source
`-o` : Ouput Path (Print out if None)
`-ot`: Output type: xml or str 

Method 0:
```
python3 main/converter.py -m 0 -i ../stock/don-thuoc.png -ot str
```
Method 1:
```
python3 main/converter.py -m 1 -i ../stock/don-thuoc.png -o test.html -ot xml
```
Method 2:
```
python3 main/converter.py -m 2 -i ../stock/don-thuoc.png -o test.html -ot xml
```

or to have more setting:  
```
python3 main/converter.py -h
```

## Online Example 
```
python online-sample/opencv-text-recognition/text_recognition.py --east online-sample/opencv-text-recognition/frozen_east_text_detection.pb --image stock/don-thuoc.png -w 320 -e 320 --padding 1
```