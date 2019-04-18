FROM python:3.5
ENV PYTHONUNBUFFERED 1
RUN mkdir /ocr
WORKDIR /ocr
COPY . /ocr/
RUN apt-get update 
RUN apt install tesseract-ocr -y 
RUN apt install tesseract-ocr-vie -y
RUN apt-get install libmagickwand-dev -y 
RUN mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xmlout
RUN pip install -r requirements.txt
