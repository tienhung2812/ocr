from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas
import os
from converter.med0.converter import Converter as M0
from converter.med1.converter import Converter as M1
from converter.med2.converter import Converter as M2


def processImage(file, method = 0, lang='vie', output_type = 'str', full_table = False):
    if method == 1:
        con = M1(file = file, lang=lang , output_type=output_type, full_table=full_table)
    elif method == 2:
        con = M2(file = file, lang=lang , output_type=output_type, full_table=full_table)
    else:
        con = M0(file = file, lang=lang , output_type=output_type, full_table=full_table)
    output,data = con.execute()
    stat = {}
    stat['mean'] = data['conf'].mean()
    stat['lang'] = lang
    stat['method'] = method
    stat['output_type'] = output_type
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
            
        result,data,stat = processImage(os.path.abspath(uploaded_file_url[1:]),method,lang,output_type,full_table)
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