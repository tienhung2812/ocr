#! /bin/bash

cd main

# docker build --tag web_image .

heroku container:push web -a hung-ocr
heroku container:release web -a hung-ocr
