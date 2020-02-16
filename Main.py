# This is a test of reading video

import cv2
import numpy as np

import backgroundSubtractionFunctions as bgf

filepath = r"D:\Programming\MATLAB\video_prog\FroggerHighway.mp4"
vid = cv2.VideoCapture(filepath)
flag = vid.isOpened()
if flag:
    print("打开摄像头成功")
else:
    print("打开摄像头失败")

ret, frame = vid.read()
w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = int(w)
h = int(h)
BG = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
Sub = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
a = 0.05
cv2.namedWindow("Background", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Subtracted", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
FrameNum = 0
InitialFrames = np.zeros((h, w, 3, 30))
# PAR = np.zeros((int(h), int(w)))
chn = 3
PAR = bgf.PixelPar((h, w, chn))
print(type(PAR))
while 1:
    FrameNum += 1
    ret, frame = vid.read()
    if FrameNum <= 30:
        InitialFrames[:, :, :, FrameNum - 1] = frame
    if FrameNum == 30:
        bgf.InitialPars(InitialFrames, PAR)
        BG =PAR.miu[:, :, 1]
    # CF = gray[:, :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    BG = cv2.addWeighted(BG, 1 - a, gray, a, 0)
    Sub = cv2.absdiff(BG, gray)
    Sub = cv2.threshold(Sub, 20, 255, type=cv2.THRESH_BINARY)
    cv2.imshow('frame', frame)
    cv2.imshow("Background", BG)
    cv2.imshow("Subtracted", Sub[1])
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
