# OCR
Vietnamese ocr for convert Scanned PDF to data

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

## Step 1: Convert Scanned PDF to xml

Folder: ```xml``` 