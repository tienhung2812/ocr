# Digitizing Receipt
Vietnamese Receipt OCR Scanner

## Presiquitive
- Docker
- docker-compose 

### Install Docker
- [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
- [Docker for MacOS](https://docs.docker.com/docker-for-mac/install/)
- [Docker for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

Following installation step from previous links to install Docker on your system.

### Install Docker-Compose
On Windows and MacOS version of Docker desktop have included Docker-compose.  

#### [Docker-compose for linux](https://docs.docker.com/compose/install/)
Run this command to download the current stable release of Docker Compose:

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
Apply executable permissions to the binary:
```
sudo chmod +x /usr/local/bin/docker-compose
```


# Run the server on your local system
```
docker-compose up
```
and go to [localhost:8000](http://localhost:8000)

## Server debugging in Visual Studio Code (VSCode)
- Make sure that server is already started
- Uncomment lines in [manage_with_debug.py](main/manage_with_debug.py) and save, server will automatically detect change in file and restart. [manage_with_debug.py](main/manage_with_debug.py) file should be like this:
```
from ocr.settings import *

DEBUG = True

import ptvsd
ptvsd.enable_attach()
print('ptvsd is started')
```
- Click on Debug button on VSCode and [add configuration](https://code.visualstudio.com/Docs/editor/debugging#_launch-configurations)
- Add following configuration in `launch.json`
```
{
    "name": "Python: Remote Attach",
    "type": "python",
    "request": "attach",
    "port": 5678,
    "host": "localhost",
    "pathMappings": [
        {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "."
        }
    ]
}
```
- Start Debugging
- Mark Breakpoint at [main/ocr_server/views.py](main/ocr_server/views.py), `image_process` function (about line 80-83)
- Go to [localhost:8000](http://localhost:8000), import image by using `Choose File` button then press `Process` to start scanning receipt. VSCode will automatically stop at marked breakpoint.

# Online Server
https://hung-ocr.herokuapp.com


## Training
### Presiquitive
- jupyter-notebook
- keras
- tensorflow

## LSTM Intent Classification Training
You can find the LSTM classification usage in [here](trainning/text_classification)

The best train at [here](trainning/text_classification/Train vi final.ipynb)

## For running the code without server
### Installation for not using serving server
Package using : [tesseract-ocr](https://github.com/tesseract-ocr/tesseract)

- Install tesseract on Ubuntu 18.04
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
pip3 install -r main/requirements.txt
```
### Run
```
main/main.py
```  
Or go to [main/main.py](main/main.py) and start Debugging normally as a Python file


# Flow chart
![alt text](graph/flowchart.png "Flow chart")



<!-- 
## **Step 1:** Convert Scanned PDF to xml (deprecated)
[Folder](main/converter)  
[Document](main/converter/README.md)  
Usage:
`-m` : Method
`-i` : Image Source
`-o` : Ouput Path (Print out if None)
`-ot`: Output type: xml or str 

Method 0:
```
python3 main/converter.py -m 0 -i ../stock/don-thuoc.png -ot str
```
Method 1:
```
python3 main/converter.py -m 1 -i ../stock/don-thuoc.png -o test.html -ot xml
```
Method 2:
```
python3 main/converter.py -m 2 -i ../stock/don-thuoc.png -o test.html -ot xml
```

or to have more setting:  
```
python3 main/converter.py -h
``` -->

