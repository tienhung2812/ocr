FROM ubuntu
RUN apt-get update && apt-get install python3 python3-pip tesseract-ocr tesseract-ocr-vie libmagickwand-dev libmysqlclient-dev -y
ENV PYTHONUNBUFFERED 1
# Set the locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xmlout
RUN pip3 install -r requirements.txt

CMD python3 manage.py runserver 0.0.0.0:$PORT