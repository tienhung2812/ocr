#starting here
import pandas as pd
import numpy as np
import cv2

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


def mergeBoxes(img,boxes,HORIZONTAL_PERCENT = 0.52, printOut = True, deleteConflict = False, getArray = False, run_fully_merge = True):
    # PERCENT SETTING
    # HORIZONTAL_PERCENT = 0.52
    if img.ndim >2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w,h = img.shape

    df = convertBoxesToPandas(boxes)
    # mergeBoxes = pd.DataFrame(None,columns=range(0,8))
    
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
        result = []
        for index, row in df.iterrows():
            if index not in added_index:
                result.append((row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
        result.extend(a)
    else:
        result = a

    # Fully merge   
    if run_fully_merge:
        print("Merged box: %d"%len(a))
        if len(a) > 0:
            fully_merge = False
            print("Boxes: %d"%len(boxes))
            print("Result: %d"%len(result))
            new_a = result
            while not fully_merge:
                old_a = np.array(new_a)
                new_a = mergeBoxes(img,old_a,HORIZONTAL_PERCENT = HORIZONTAL_PERCENT,printOut = True, deleteConflict=True,getArray = True, run_fully_merge=True)
                print("old_a: %d" %len(old_a))
                print("new_a: %d" %len(new_a))
                if len(old_a) == len(new_a):
                    fully_merge = True
                    result = new_a
        
        # while not fully_merge:
        #     old_a = np.array(result)
        #     new_a = mergeBoxes(img,old_a,HORIZONTAL_PERCENT = HORIZONTAL_PERCENT,printOut = False, deleteConflict=True,getArray = True)
        #     print("Fully Merge")
        #     print(len(old_a))
        #     print(len(new_a))
        #     if len(new_a) == len(old_a):
        #         fully_merge = True


    # if not getArray:
    return np.array(result)
    # else:
        # return result

 

