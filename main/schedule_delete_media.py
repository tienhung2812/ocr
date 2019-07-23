# coding: utf-8

import os
import shutil
import time
import datetime
import django
import logging
import schedule
from ocr.settings import SCHEDULE_DELETE_MEDIA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ocr.settings")
django.setup()


def delete_media_file():
    logger.info("Checking at %s"%str(datetime.datetime.now()))
    folder = os.getcwd() + '/media'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            else:
                shutil.rmtree(file_path)
            logger.info("Deleted: %s"%str(file_path))
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            logger.error(e)

if __name__ == "__main__":
    schedule.every(SCHEDULE_DELETE_MEDIA).minutes.do(delete_media_file)

    while True:
        schedule.run_pending()
        time.sleep(10)
