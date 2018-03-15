import sys
import time
import cv2
import numpy as np
 
'''
Note: there is a bug somewhere in the MSER part (in my code I suppose).
If you play with different threshold values you'll find a few (127, for the 
image I'm using now) where the MSER bounding boxes make no sense. While 
with 115 and 135, for example, it works fine.
'''
 
# I'm interested in grouping letters into text so I group only elements that are
# close and on the same line. Change this to allow freeform groupings.
def is_close(rect1,rect2):
 
    max_x1, min_x1, max_y1, min_y1 = rect1[0]+rect1[2],rect1[0],rect1[1]+rect1[3],rect1[1]
    max_x2, min_x2, max_y2, min_y2 = rect2[0]+rect2[2],rect2[0],rect2[1]+rect2[3],rect2[1]
 
    if intersection(rect1, rect2):
        return True
 
    # how far horizontally are the boxes
    xdiff = abs(max(min_x1,min_x2) - min(max_x1, max_x2))
 
    # how far vertically are the centroids of the boxes
    mean_y1=(max_y1+min_y1) / 2.
    mean_y2=(max_y2+min_y2) / 2.
    ydiff = abs(mean_y1 - mean_y2)
 
    # we use two different thresholds
    if xdiff < 50 and ydiff < 15:
        return True
    return False
 
# A rect is not a contour so let's keep both representations (contours are
# easier to draw)
def bounding_rect_to_contour(c):
    (x,y,w,h) = cv2.boundingRect(c)
    return ( (x,y,w,h), np.asarray([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]]) )
 
# simple (faster) true/false intersection check
def intersection(r1,r2):
    X,Y,A,B = r1
    X1,Y1,A1,B1 = r2
    return not (A<X1 or A1<X or B<Y1 or B1<Y)
 
# here we want the rect
def intersection_rect(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return ()
  return (x, y, w, h)
 
# the real deal
def find_words(img):
 
    # convert to gray and binary threshold
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,115,255,0)
 
    # let's find the contours
    mser=True
    if mser:
        # use this to play with MSER params: 
        # https://docs.opencv.org/trunk/d3/d28/classcv_1_1MSER.html#a136c6b29fbcdd783a1003bf429fa17ed
        #mser = cv2.MSER_create(5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5)
 
        mser = cv2.MSER_create()
        regions = mser.detectRegions(thresh)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
        contours = hulls
    else:
        thresh = cv2.bitwise_not(thresh) # wants black bg
        im2,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
 
 
    # Found their bounding boxes (as rectangles and as contours, just to make life easier later)
    # and put them all together
    bboxes = []
    for i,c in enumerate(contours):
        rect, cr = bounding_rect_to_contour(c)
        bboxes.append((cr,rect,c))
 
    contours = bboxes
 
    '''
    base_contours=list(c[0] for c in bboxes)
    cv2.drawContours(img, base_contours, -1, (0,0,255), 1)
    cv2.imshow('base contours', img)
    cv2.waitKey(0)
    '''
 
    # compare each box to each other and, if they are close, assign them the same group number
    # Each elements in status correspond to a box and the value is the group it belongs to
    LENGTH = len(contours)
    status = np.zeros(LENGTH)
    for i,cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                if is_close(cnt1[1],cnt2[1]):
                    val = min(status[i],status[x])
                    all_boxes_in_this_group = np.where(status == status[i])
                    status[all_boxes_in_this_group] = val
                    status[x] = val
                else:
                    if status[x]==status[i]:  
                        status[x] = i+1         # let's start a new group
 
    # let's merge all the boxes from the same group together
    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i][2] for i in pos)  # here we finally use the actual contour
            hull = cv2.convexHull(cont)
            unified.append((hull, cv2.boundingRect(hull)))  # and let's prepare the rect too
 
    # now discard boxes that are fully contained inside larger ones
    child_status = np.zeros(len(unified), dtype=np.int)
    for i,cnt1 in enumerate(unified):
        is_child = False
        for j,cnt2 in enumerate(unified[i+1:]):
            rect1 = cnt1[1]
            rect2 = cnt2[1]
            intersect = intersection_rect(rect1, rect2)
            if rect1 == intersect:
                child_status[i] = 1
                break
            elif rect2 == intersect:
                child_status[j+i+1] = 1
 
    # 1 means child boxes, so keep the zeros only
    unified = np.asarray(unified)[np.where(child_status == 0)]
 
    return unified
 
# discard everythink except what's inside the contours
# In pratice may remove some "dust" just outside the contours
# and the fragments ignored by the MSER
def applyMask(img, contours):
 
    # grayscale and negate
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
 
    dim = np.shape(img)
 
    mask = np.zeros((dim[0], dim[1]), dtype=np.float)
    CV_FILLED=-1
    cv2.drawContours(mask, contours, -1, (1,1,1), CV_FILLED)
    # add some border otherwise will cut 1px inside the box
    cv2.drawContours(mask, contours, -1, (1,1,1), 2)
 
    # mask is 0,1 so just multiply to set everything to zero or to the current value
    masked = img * (mask.astype(img.dtype))
 
    # flip colors back
    return cv2.bitwise_not(masked)
 
if __name__ == "__main__":
 
    start= time.time()
 
    # let's read the image (full color mode)
    img = cv2.imread('TestCases/test10.jpg')
 
    unified = find_words(img)
 
    print("Elapsed: ", (time.time()-start), "s")  # 0.078s for an 800x600 img
 
    # extract the contours
    unified_cnt=list(c[0] for c in unified)
 
    # display the result
    draw_img = img.copy()
    cv2.drawContours(draw_img,unified_cnt,-1,(0,255,0),2)   #green
    for c in unified:
        (x,y,w,h) = c[1]
        cv2.rectangle(draw_img, (x,y), (x+w,y+h), (255, 0, 0), 1)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)
 
    # optionally apply a mask
    result = applyMask(img, unified_cnt)
    cv2.imwrite("result.png", result);
 
    cv2.destroyAllWindows()
