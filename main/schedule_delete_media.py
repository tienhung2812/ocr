# coding: utf-8

import os
import shutil
import time
import datetime
import django

import schedule
from ocr.settings import SCHEDULE_DELETE_MEDIA

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ocr.settings")
django.setup()


def delete_media_file():
    print("Checking at ",datetime.datetime.now())
    folder = os.getcwd() + '/media'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            else:
                shutil.rmtree(file_path)
            print("Deleted: ",file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    schedule.every(SCHEDULE_DELETE_MEDIA).minutes.do(delete_media_file)

    while True:
        schedule.run_pending()
        time.sleep(10)
