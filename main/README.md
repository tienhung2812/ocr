#Step to deploy to heroku

Append 
```
CMD python manage.py runserver 0.0.0.0:$PORT
```
in Dockerfile

```
docker build --tag web_image .
heroku container:push web -a hung-ocr
heroku container:release web -a hung-ocr
```

Link: https://github.com/twtrubiks/Deploying_Django_To_Heroku_With_Docker