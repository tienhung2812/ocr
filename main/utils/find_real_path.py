import os

def get_real_path(path):
    if path[0] == '/':
        path = path[1:]
    return os.path.abspath(path)

def get_url_path(path):
    current_path = os.getcwd()
    return path.replace(current_path,"")