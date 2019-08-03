# Text Classification

### Presiquitive

```
python3
pip3
```

Requirement libs
```
pandas
numpy
pickle
keras
tensorflow
opencv-python
matplotlib
sklearn
jupyter
```

### Run jupyter notebook
```
jupyter notebook
```

## Notebooks
### Process data
This is process data from `ocr_server` to csv form

### Divide data
On the csv first from, there are mix up between English and Vietnamese. This is for dividing the languages data

### Trainning on sequence only
This is the first model, using 9 different layers

title, date, phone_num,	address,	index, content, total, brand_name,	thank_you

### Training on sequence and location
Model using location **X,Y,W,L, area** to enhance accuracy

### Training on sequence and grouped col
Cut off the number of layers to 
info, index, content, total, brand_name,thank_you

### Training final vi
This is the final training uses on the website, this models only uses sentence to classify fields of Vietnamese receipt into 6 layers: info, index, content, total, brand_name,thank_you