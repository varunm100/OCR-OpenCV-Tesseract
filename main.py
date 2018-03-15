from PIL import Image
from scipy.misc import imsave
import numpy as np
import pytesseract
import cv2 as cv
import doc2text

def binarizeGuasianImage(inputFile):
	img = cv.imread(inputFile, 0);
	return cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

def GetMSER(inputFile):
	img = cv.imread(inputFile);
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	vis = img.copy()
	mser = cv.MSER_create()
	regions, _ = mser.detectRegions(gray)
	hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv.polylines(vis, hulls, 1, (0, 255, 0))
	cv.imshow('Image View', vis)
	cv.waitKey(0)
	mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
	for contour in hulls:
		cv.drawContours(mask, [contour], -1, (255, 255, 255), -1)
	textOnly = cv.bitwise_and(img, img, mask=mask)
	cv.imshow("Text View", textOnly)
	cv.waitKey(0)
	cv.destroyAllWindows()

def Doc2Text(inputPath):
	doc = doc2text.Document()
	doc = doc2text.Document(lang="eng")
	doc.read(inputPath)
	doc.process()
	doc.extract_text()
	text = doc.get_text()
	print(text)

#cv.imwrite('final.png',binarizeGuasianImage('TestCases/test7.jpeg'));
GetMSER('TestCases/test2.jpeg')
#Doc2Text('TestCases/test1.jpg')
