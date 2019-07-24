import calendar
import os
import time

import pandas

from converter.med0.converter import Converter as M0
from converter.med1.converter import Converter as M1
from converter.med2.converter import Converter as M2
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.template import loader
from image.process import ReceiptImage
from text_detection.core.detector import TextDetection
from text_recognization.text_recognizance import TextRecognizance
from text_combination.text_combine import TextCombinator
from utils.find_real_path import *

def processImage(file, method = 0, lang='vie', output_type = 'str', full_table = False, config='--oem 1'):
    if method == 1:
        con = M1(file = file, lang=lang , output_type=output_type, full_table=full_table ,config = config)
    elif method == 2:
        con = M2(file = file, lang=lang , output_type=output_type, full_table=full_table,config = config)
    else:
        con = M0(file = file, lang=lang , output_type=output_type, full_table=full_table,config = config)
    output,data = con.execute()
    stat = {}
    stat['mean'] = data['conf'].mean()
    stat['lang'] = lang
    stat['method'] = method
    stat['output_type'] = output_type
    stat['conf'] = config
    # print(output)
    return output,data,stat


def index(request):
    col_data='col-6'
    if request.method == 'POST' and request.FILES['myfile'] and request.POST['method']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        method = request.POST['method']
        output_type = request.POST['output']
        lang = request.POST['lang']
        full_table = request.POST.get('fulltable',False)
        if full_table != False:
            full_table = True
            col_data='col-12'

        oem = request.POST['oem']
        conf = '--oem '+ oem
        psm = request.POST['psm']
        if psm != ' None':
            conf += ' --psm '+ psm


        result,data,stat = processImage(os.path.abspath(uploaded_file_url[1:]),method,lang,output_type,full_table, conf)
        return render(request, 'ocr_server/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'result': result,
            'data_result': data.to_html(),
            'col_data':col_data,
            'stat':stat
        })

    template = loader.get_template('ocr_server/index.html')
    context = {
        'uploaded_file_url': "http://www.tourniagara.com/wp-content/uploads/2014/10/default-img.gif",
        'col_data':col_data
    }
    return HttpResponse(template.render(context, request))

def image_process(request):
    col_data='col-6'
    if request.method == 'POST' and request.FILES['myfile']:
        #Create TRANSACTIONNUM
        TRANSACTION_NUM = str(calendar.timegm(time.gmtime()))

        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(TRANSACTION_NUM+'/'+myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        path = get_real_path(uploaded_file_url)

        filename = os.path.basename(uploaded_file_url)
        ri = ReceiptImage(path, filename,TRANSACTION_NUM)
        ri.processImage()

        # Detect line
        td = TextDetection(ri.wraped_url,TRANSACTION_NUM)
        text_line_file_image_url,final_text_line_file_image_url,text_line_file_box_url, cropped_image_array = td.find()
        
        # tr = TextRecognizance(cropped_image_array)

        # text_result = tr.detect_array()
        
        # groupResult = []
        # for i in range(0,len(cropped_image_array)):
        #     groupResult.append({
        #         "image":cropped_image_array[i],
        #         "text":text_result[i]['text'],
        #         "conf": text_result[i]['conf'],
        #     })
        # # Croper image
        # cropped_image_array = ri.transformImage(text_line_file_box_url)
        
        # cropped_image_array_by_box = ri.transformImageByBoxes(boxes)

        # result,data,stat = processImage(os.path.abspath(uploaded_file_url[1:]),method,lang,output_type,full_table, conf)
        return render(request, 'ocr_server/image_process.html', {
            'uploaded_file_url': uploaded_file_url,
            'dilated_file_url':get_url_path(ri.dilated_url),
            'drawed_file_url': get_url_path(ri.drawed_url),
            'cropped_file_url': get_url_path(ri.wraped_url),
            'final_text_line_file_image_url':get_url_path(final_text_line_file_image_url),
            'cropped_image_array': cropped_image_array,
            'transaction_num': TRANSACTION_NUM,
            'text_line_file_url':get_url_path(text_line_file_image_url),
            'status':ri.status 
        })

    template = loader.get_template('ocr_server/image_process.html')
    context = {
        'uploaded_file_url': "http://www.tourniagara.com/wp-content/uploads/2014/10/default-img.gif",
 
    }
    return HttpResponse(template.render(context, request))


def text_recognization(request):
    if request.method == 'POST':
        tr = TextRecognizance(image_url=request.POST.get('url'),transaction_num = request.POST.get('transaction'),id_num = request.POST.get('id'))

        text_result = tr.detect_image()
        return JsonResponse(text_result)
    else:
        return HttpResponseRedirect("/")

def text_combine(request):
    if request.method == 'POST':
        text_result = {"ok":False}
        transaction_num = request.POST.get('transaction')
        input_image = request.POST.get('input_image')
        if len(transaction_num) > 0:
            tc = TextCombinator(transaction_num,input_image)
            text_result = {"ok":True}
        return JsonResponse(text_result)
    else:
        return HttpResponseRedirect("/")

