#starting here
import pandas as pd
import numpy as np

merged_str = """160,61,496,61,496,105,160,105,0.9995301,0
112,205,464,205,464,235,112,235,0.9995153,1
176,232,432,232,432,263,176,263,0.99949753,2
64,260,512,260,512,290,64,290,0.99949694,3
80,286,512,286,512,318,80,318,0.9994821,4
80,314,512,314,512,345,80,345,0.999471,5
144,342,432,342,432,372,144,372,0.9994646,6
240,371,352,371,352,396,240,396,0.9994591,7
16,420,464,420,464,454,16,454,0.99944884,8
16,447,208,447,208,475,16,475,0.9994455,9
16,473,544,473,544,509,16,509,0.99944395,10
16,502,496,502,496,537,16,537,0.9994417,11
16,529,192,529,192,556,16,556,0.9994399,12
16,556,544,556,544,592,16,592,0.9994374,13
16,583,432,583,432,616,16,616,0.9994326,14
16,611,208,611,208,638,16,638,0.99942386,15
16,638,544,638,544,671,16,671,0.99942017,16
16,664,368,664,368,693,16,693,0.9994165,17
16,691,176,691,176,716,16,716,0.99941456,18
16,718,544,718,544,752,16,752,0.9994005,19
16,772,576,772,576,810,16,810,0.9993974,20
16,823,96,823,96,848,16,848,0.9993949,21
432,836,560,836,560,861,432,861,0.9993944,22
16,852,560,852,560,888,16,888,0.9993917,23
0,876,560,876,560,941,0,941,0.99938464,24
16,936,560,936,560,969,16,969,0.9993801,25
0,985,560,985,560,1021,0,1021,0.9993754,26
0,1014,352,1014,352,1041,0,1041,0.999371,27
80,1067,464,1067,464,1099,80,1099,0.9993685,28
144,1096,432,1096,432,1126,144,1126,0.9993643,29
112,1122,448,1122,448,1152,112,1152,0.9993618,30"""

default_str = """160,61,496,61,496,105,160,105,0.9995301,0
112,205,464,205,464,235,112,235,0.9995153,1
176,232,432,232,432,263,176,263,0.99949753,2
64,260,512,260,512,290,64,290,0.99949694,3
80,286,512,286,512,318,80,318,0.9994821,4
80,314,512,314,512,345,80,345,0.999471,5
144,342,432,342,432,372,144,372,0.9994646,6
240,371,352,371,352,396,240,396,0.9994591,7
16,420,464,420,464,454,16,454,0.99944884,8
16,447,208,447,208,475,16,475,0.9994455,9
16,473,144,473,144,501,16,501,0.99944395,10
320,482,544,482,544,509,320,509,0.9994417,11
16,502,496,502,496,537,16,537,0.9994399,12
16,529,192,529,192,556,16,556,0.9994374,13
16,556,128,556,128,582,16,582,0.9994326,14
320,562,544,562,544,592,320,592,0.99942386,15
16,583,432,583,432,616,16,616,0.99942017,16
16,611,208,611,208,638,16,638,0.9994165,17
16,638,208,638,208,665,16,665,0.99941456,18
336,643,544,643,544,671,336,671,0.9994005,19
16,664,368,664,368,693,16,693,0.9993974,20
16,691,176,691,176,716,16,716,0.9993949,21
16,718,208,718,208,744,16,744,0.9993944,22
304,723,544,723,544,752,304,752,0.9993917,23
16,772,192,772,192,798,16,798,0.99938464,24
416,780,576,780,576,810,416,810,0.9993801,25
16,823,96,823,96,848,16,848,0.9993754,26
432,836,560,836,560,861,432,861,0.999371,27
16,852,272,852,272,881,16,881,0.9993685,28
448,862,560,862,560,888,448,888,0.9993643,29
272,876,400,876,400,936,272,936,0.9993618,30
16,879,208,879,208,905,16,905,0.9993568,31
432,885,560,885,560,916,432,916,0.9993556,32
0,903,96,903,96,929,0,929,0.99935395,33
432,915,560,915,560,941,432,941,0.99934715,34
16,936,112,936,112,955,16,955,0.99933726,35
432,943,560,943,560,969,432,969,0.9993338,36
0,985,560,985,560,1021,0,1021,0.99933237,37
0,1014,352,1014,352,1041,0,1041,0.9993291,38
80,1067,464,1067,464,1099,80,1099,0.9993192,39
144,1096,432,1096,432,1126,144,1126,0.99931586,40
112,1122,448,1122,448,1152,112,1152,0.99931335,41"""

def getBoxes():
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
    return boxes

def convertBoxesToPandas(boxes):
    numpy_array = np.array(boxes)
    df = pd.DataFrame(data=numpy_array,dtype=np.int)
    return df

def sortBoxes(boxes):
    #Convert to data frame
    df = convertBoxesToPandas(boxes)
    df = df.sort_values(by=[1,0])
    print(df)
    boxes = df.to_numpy()
    return boxes


def skew():
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

def drawBox(img,boxes, color=(0,255,0),puttext=True):
    for i, box in enumerate(boxes):
        if puttext:
            cv2.putText(img,str(i),(box[6],box[7]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                    thickness=2)
    return img

def getLeftVerticalColumn(row):
    return row[1], row[7]

def getRightVerticalColumn(row):
    return row[3], row[5]

def getMaxBoudingBox(w,col):
    return 0,col[0], w,col[0],w,col[1],0,col[1]

def evaluteCrossingBox(col1,col2):
    """[GRAPH]
    a
    |
    c
    |
    b
    |
    d

    b-c / c-a
    """
    # First check is crossed
    if col2[0] > col1[1] or col1[0]>col2[1]:
        return -1 

    # A & C: smaller is A
    if col1[0] > col2[0]:
        a = col2[0]
        c = col1[0]
    else:
        c = col2[0]
        a = col1[0]

    # B & D: smaller is B
    if col1[1] <= col2[1]:
        b = col1[1]
        d = col2[1]
    else:
        d = col1[1]
        b = col2[1]

    lenght_col1 = abs(col1[1]-col1[0])
    lenght_col2 = abs(col2[1]-col2[0])

    # apperance percent of col2 on col1
    case1_2o1 = float(abs(c-b))/float(lenght_col1)
    # apperance percent of col1 on col2
    case1_1o2 = abs(c-b)/lenght_col2

    # print("a: %d, b: %d, c: %d, d: %d" %(a,b,c,d))
    # print("(%d-%d)/(%d)"%(c,b,lenght_col1))
    # print("result: %f"%(case1_2o1))
    # print("col1[0]: %d, col1[1]: %d, col2[0]: %d, col2[1]: %d" %(col1[0],col1[1],col2[0],col2[1]))


    return case1_2o1

    """[Case 2]
    """

def mergeTwoBoxes(box1,box2):
    numpy_array = np.array([box1,box2])
    df = pd.DataFrame(data=numpy_array,dtype=np.int)

    left = df[0].min()
    right = df[2].max()
    top = df[1].min()
    bottom = df[5].max()
    return left,top,right,top,right,bottom,left,bottom


def mergeBoxes(img,boxes, printOut = True, deleteConflict = False):
    # PERCENT SETTING
    HORIZONTAL_PERCENT = 0.52

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w,h = img.shape

    df = convertBoxesToPandas(boxes)
    mergeBoxes = pd.DataFrame(None,columns=range(0,8))
    
    #SETTING MERGE COLUMN
    merge_column = 1,3
    a = []
    # Add result
    added_index = []
    for index, row in df.iterrows():
        #Get the left vertical column
        y1, y2 = getLeftVerticalColumn(row)

        #Get most left and most right bounding box from this y1,y2
        boudingBox = getMaxBoudingBox(w,(y1,y2))

        # Evaluate the cross of other box in this section
        for index_e, row_e in df.iterrows():
            if index_e == index:
                continue
            else:
                col2_y1, col2_y2 = getLeftVerticalColumn(row_e)
                col1 = (y1,y2)
                col2 = (col2_y1, col2_y2)
                
                cross_percent = evaluteCrossingBox(col1,col2)
                if cross_percent >= 0:
                    if printOut:
                        # print("col1[0]: %d, col1[1]: %d, col2[0]: %d, col2[1]: %d" %(col1[0],col1[1],col2[0],col2[1]))
                        print('===========')
                        print('Evaluate: ',end='')
                        print(index)
                        print('Value: ', end='')
                        print(getLeftVerticalColumn(row))
                        print("Compare with: ",end='')
                        print(index_e)
                        print('Value: ', end='')
                        print(getLeftVerticalColumn(row_e))                    
                        print('Percent: ', end='')
                        print(cross_percent)

                    if cross_percent >= HORIZONTAL_PERCENT:
                        if mergeTwoBoxes(row.values,row_e.values) not in a:
                            a.append(mergeTwoBoxes(row.values,row_e.values))   
                        if index_e not in added_index:
                            added_index.append(index_e)

    if deleteConflict:
        for index, row in df.iterrows():
            if index not in added_index:
                a.append(row)
    return np.array(a)

def getCheckVar(i,box,merged_box):
    i*=2
    return box[i],box[i+1],merged_box[i],merged_box[i+1]

def checkIsInBoxes(box,merged_box,not_same_box = False):
    """
    a--b
    |  |
    d--c
    
    0--1
    |  |
    3--2

    x1,y1: box pt
    x2,y2: merged_box pt
    """
    if not_same_box:
        is_same_box = True
        for i in range(0,len(box)):
            if box[i] != merged_box[i]:
                is_same_box = False
                break

        if is_same_box:
            return False

    # Init variable
    a = False
    b = False
    c = False
    d = False

    # Top left: i = 0, a
    x1, y1, x2, y2 = getCheckVar(0,box,merged_box)
    a = (x2 <= x1) and (y2 <= y1)

    # Top right: i = 1, b
    x1, y1, x2, y2 = getCheckVar(1,box,merged_box)
    b = (x2 >= x1) and (y2 <= y1)

    # Bottom right: i = 2, c
    x1, y1, x2, y2 = getCheckVar(2,box,merged_box)
    c = (x2 >= x1) and (y2 >= y1)

    # Bottom left: i = 3, d
    x1, y1, x2, y2 = getCheckVar(3,box,merged_box)
    d = (x2 <= x1) and (y2 >= y1)

    return (a and b and c and d)

def removeConflictBoxes(boxes,merged_boxes):
    boxes = pd.DataFrame(data=boxes,dtype=np.int) 
    # print(merged_boxes)
    merged_boxes = pd.DataFrame(data=merged_boxes,dtype=np.int)  

    result = []
    added_index = []
    for index, box in boxes.iterrows():
        add_box = None
        for index_e, merged_box in merged_boxes.iterrows():
            if checkIsInBoxes(box.values,merged_box.values):
                add_box = merged_box
                added_index.append(index)
            elif index not in added_index:
                add_box = box

            if add_box:
                add_box = add_box.tolist()
                if add_box not in result:
                    result.append(add_box)
    
    return np.array(result)


def convert_str_to_boxes_array(default_str):
    a = default_str.split('\n')


    boxes = []
    i = 0
    j = 0
    for row in a:
        j=0
        cont = row.split(',')
        boxes.append([])
        for col in cont:
            t = None
            if '.' in col:
                t = float(col)
            else:
                t = int(col) 
            boxes[i].append(t)
            j+=1
            if j==8:
                break
        i+=1

    return boxes

def getPandasWithWrapped(default_boxes, merged_boxes, score = None):
    input_index = ['x_tl','y_tl','x_tr','y_tr','x_bl','y_bl','x_br','y_br']
    output_index = ['x_tl','y_tl','x_tr','y_tr','x_bl','y_bl','x_br','y_br', 'score','parrent']
    # boxes_df = pd.DataFrame(default_boxes,columns=input_index)
    # merged_boxes_df = pd.DataFrame(merged_boxes,columns=input_index)
    
    data = pd.DataFrame(columns=output_index)
    box_have_child = []
    child_box = []
    score_i = 0
    for mbid, merged_box in enumerate(merged_boxes):
        have_score = True
        for i, box in enumerate(default_boxes):
            if (checkIsInBoxes(box,merged_box,not_same_box = True)):
                cont = box
                if score:
                    cont.append(score[i])
                else:
                    cont.append(-1)
                cont.append(mbid)
                child_box.append(cont)

                have_score = False
                if mbid not in box_have_child:
                    box_have_child.append(mbid)
        box_score = -1
        if score and have_score:
            box_score = score[score_i]
            score_i+=1
        
        # Add to data
        cont = merged_box
        cont.append(box_score)
        cont.append(-1)
        data.loc[mbid] = cont
    
    # Add child box to data
    for i, box in enumerate(child_box):
        locat = len(data)+i
        data.loc[locat] = box

    return data





boxes = convert_str_to_boxes_array(default_str)
merged_boxes = convert_str_to_boxes_array(merged_str)
print(len(boxes))
print(len(merged_boxes))
result = getPandasWithWrapped(boxes, merged_boxes)

# import cv2
# boxes = getBoxes()
# boxes = sortBoxes(boxes)
# img = cv2.imread('original_image.png')
# img = drawBox(img,boxes)
# merge_box = mergeBoxes(img,boxes)
# img = drawBox(img,merge_box, color=(255,0,0), puttext = False)

# merge_box = mergeBoxes(img,boxes,deleteConflict=True)
# img = drawBox(img,merge_box, color=(255,0,0), puttext = False)



# cv2.imshow('image',img)
# cv2.waitKey(0)

