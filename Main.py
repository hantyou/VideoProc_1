# This is a test of reading video

import cv2
import numpy as np

import backgroundSubtractionFunctions as bgf

filepath = r"D:\Programming\MATLAB\video_prog\MVI_1738.MOV"
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
print((w, h))
BG = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
Sub = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
cv2.namedWindow("Background_Show_Simple_Update", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Subtracted", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow(" SBG_Gray",cv2.WINDOW_KEEPRATIO)
a = 0.0025  # 简单更新的更新率
FrameNum = 0  # 初始化帧计数器
InitialFrN = 20  # 设置初始帧帧数
chn = 4
InitialFrames = np.zeros((h, w, chn, InitialFrN),dtype=np.float32)  # 设置一个4维数组存放InitialFrN个三维色彩图像
print('======正在创建参数存储区======')
PAR = bgf.PixelPar((h, w, chn))
print('======创建完毕======')
"""
    PAR.xxxx[chn, y, x, nModel]的四个参数含义分别为
    chn: 当前色彩通道，分别对应B，G，R；
    y，x分别为高和宽；
    nModel：高斯分量的选择；
    ——（只要在Main中一直选择0分量，那么只要在更新程序中一直将背景分量排序到第一就可以了）；
"""
# print(type(PAR))
BG_Show = BG
FrameInNdarray = np.array([[[[0] * 3] * w] * h] * chn, dtype=np.float32)
while 1:
    """======帧计数器======"""
    FrameNum += 1
    """======获取当前帧======"""
    ret, ColorFrame = vid.read()
    """======创建色彩通道分开的帧======"""
    GrayFrame = cv2.cvtColor(ColorFrame, cv2.COLOR_BGR2GRAY)
    [B, G, R] = cv2.split(ColorFrame)
    SelfFrame = [np.float32(B), np.float32(G), np.float32(R), np.float32(GrayFrame)]
    """======初始化部分======"""
    if FrameNum < InitialFrN:  # 记录前30帧
        for i in range(chn):
            InitialFrames[:, :, i, FrameNum - 1] = SelfFrame[i]
        continue
    if FrameNum == InitialFrN:  # 用前InitialFrN帧的平均作初始化
        bgf.InitialPars(InitialFrames, PAR)
        BG_Show = np.uint8(PAR.miu[1, :, :, 0])
        """定义彩色背景"""
        BG_B = PAR.miu[0, :, :, 0]
        BG_G = PAR.miu[1, :, :, 0]
        BG_R = PAR.miu[2, :, :, 0]
        BG_Gray = PAR.miu[3, :, :, 0]
        """-----------"""
        ColorBG = cv2.merge([BG_B, BG_G, BG_R])
        cv2.namedWindow("Color Background", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Color Background", np.uint8(ColorBG))
        BG = [BG_B, BG_G, BG_R, BG_Gray, Sub]
        PAR.ShowPar(x=639, y=479, chn=0)
        PAR.ShowPar(x=639, y=479, chn=1)
        PAR.ShowPar(x=639, y=479, chn=2)
    """======初始化结束======"""
    #BG_B = cv2.addWeighted(np.uint8(BG_B), 1 - a, B, a, 0)
    #BG_G = cv2.addWeighted(np.uint8(BG_G), 1 - a, G, a, 0)
    #BG_R = cv2.addWeighted(np.uint8(BG_R), 1 - a, R, a, 0)
    ColorBG = cv2.merge([BG_B, BG_G, BG_R])
    cv2.imshow("Color Background", np.uint8(ColorBG))

    for i in range(len(SelfFrame)):
        for j in range(3):
            FrameInNdarray[i, :, :, j] = np.float32(SelfFrame[i])
    """======升级背景======"""
    BG_Show = cv2.addWeighted(BG_Show, 1 - a, GrayFrame, a, 0)
    bgf.updateBackGround(BG=BG, frame=SelfFrame, FrameInNdarray=FrameInNdarray, PAR=PAR)
    [SBG_B, SBG_G, SBG_R, SBG_Gray]=[np.uint8(BG[0]), np.uint8(BG[1]), np.uint8(BG[2]), np.uint8(BG[3])]
    SBG=[SBG_B, SBG_G, SBG_R, SBG_Gray]
    """======执行背景相减步骤======"""
    DirectSub = cv2.absdiff(BG_Show, GrayFrame)
    DirectSub2 = cv2.absdiff(SBG[3], GrayFrame)
    # Sub = bgf.GetSub(BG,frame)
    """======执行二值化步骤======"""
    [threshold, Sub] = cv2.threshold(DirectSub2, 20, 255, type=cv2.THRESH_BINARY)
    """======显示计算结果======"""
    # print(PAR.weight[1,34, 1, :] )
    cv2.imshow('frame', ColorFrame)
    cv2.imshow("Background_Show_Simple_Update", BG_Show)
    cv2.imshow("Subtracted", Sub)
    cv2.imshow(" SBG_Gray", SBG_Gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
