FROM ubuntu
RUN apt-get update && apt-get install python3 python3-pip tesseract-ocr tesseract-ocr-vie libmagickwand-dev libmysqlclient-dev curl unzip -y
ENV PYTHONUNBUFFERED 1
# Set the locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xmlout
RUN pip3 install -r requirements.txt

# Download CTPN Checkpoint
RUN chmod 700 gdrive.sh
RUN curl gdrive.sh | bash -s https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view

RUN mv checkpoints_mlt.zip  text_detection/
RUN unzip checkpoints_mlt.zip

# COMPILE 
RUN chmod +x text_detection/utils/bbox/make.sh
RUN text_detection/utils/bbox/make.sh
RUN text_detection/utils/bbox/ -R 

CMD python3 manage.py runserver 0.0.0.0:$PORT