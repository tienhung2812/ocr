# pull in the normal settings
from ocr.settings import *

# debug for us
DEBUG = True

import ptvsd
ptvsd.enable_attach()
print('ptvsd is started')