# OCR
Vietnamese ocr for convert Scanned PDF to data

## Step 1: Convert Scanned PDF to xml
Package using : [tesseract-ocr](https://github.com/tesseract-ocr/tesseract)

- Install tesseract on linux 18
```
sudo apt install tesseract-ocr
```

- Install Vietnamese languagues data pack from [VietUnicode](http://vietunicode.sourceforge.net/howto/tesseract-ocr.html)

```
tesseract vietsample.tif output –l vie
```