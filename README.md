OCR-OpenCV-Tesseract

g++ *.cpp -std=gnu++11 -o OCR-OpenCV-Tesseract -llept -ltesseract `pkg-config --cflags --libs opencv` <-- To compile <--

OCR tesseract text recognition using various pre-processing methods such as, deskewing, forground detection, and adaptive thresholding. 
