file = open("sample.txt", "r") 
boxes = []
for line in file.readlines():
    # print(line)
    # print(line.split(','))
    a = []
    for thing in line.split(','):
        if thing == '\n':
            continue
        else:
            num = float(thing)
            if round(num) == num:
                a.append(int(num))
    
    boxes.append(a)

#starting here
import pandas as pd
import numpy as np

#Convert to data frame
numpy_array = np.array(boxes)
df = pd.DataFrame(data=numpy_array,dtype=np.int)
df = df.sort_values(by=[1,0])
print(df)
boxes = df.to_numpy()

import cv2

img = cv2.imread('original_image.png')

#Skew

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
afterthresh = cv2.bitwise_not(thresh)
coords = np.column_stack(np.where(thresh > 0))
angle = 1

if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", img)
cv2.imshow("thresh", afterthresh)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)


for i, box in enumerate(boxes):
    cv2.putText(img,str(i),(box[6],box[7]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                thickness=2)

cv2.imshow('image',img)
cv2.waitKey(0)