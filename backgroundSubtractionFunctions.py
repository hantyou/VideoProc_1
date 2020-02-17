import math

import numpy as np


class PixelPar:
    def __init__(self, size):
        size = list(size)
        k = 0.5
        [h, w, chn] = size
        self.weight = np.array([[[[1 / 3] * 3] * w] * h] * chn)
        self.miu = np.array([[[[0] * 3] * w] * h] * chn)
        self.sigma = np.array([[[[1] * 3] * w] * h] * chn)
        self.N = np.array([[[[1] * 3] * w] * h] * chn)
        self.M = np.array([[[[1] * 3] * w] * h] * chn)
        self.Z = np.array([[[[1] * 3] * w] * h] * chn)
        self.nGM = np.array([[[3] * w] * h] * chn)

    def ShowPar(self, x, y, chn):
        weight = self.weight[chn, y, x, :]
        miu = self.miu[chn, y, x, :]
        sigma = self.sigma[chn, y, x, :]
        N = self.N[chn, y, x, :]
        M = self.M[chn, y, x, :]
        Z = self.Z[chn, y, x, :]
        n = self.nGM[chn, y, x]
        c = ["B", "G", "R"]
        print("权重：", str(weight))
        print("均值：", str(miu))
        print("方差：", str(sigma))
        print("N: ", str(N))
        print("M: ", str(M))
        print("Z: ", str(Z))
        print("当前色彩通道: ", c[chn])
        """
        self.miu = np.zeros(shape=size, dtype=list)
        self.sigma = np.zeros(shape=size, dtype=list)
        self.nGM = np.ones(shape=size, dtype=list)
        """


class SinglePixelPar:
    def __init__(self):
        self.weight = [1 / 3, 1 / 3, 1 / 3]
        self.miu = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.nGM = 1


"""
    PAR.xxxx[chn, y, x, nModel]的四个参数含义分别为
    chn: 当前色彩通道，分别对应B，G，R；
    y，x分别为高和宽；
    nModel：高斯分量的选择；
    ——（只要在Main中一直选择0分量，那么只要在更新程序中一直将背景分量排序到第一就可以了）；
"""


def updateBackGround(BG, frame, FrameInNdarray, PAR, ControlPar='控制参数暂时为空'):
    def UpDateWithoutNowBG(frame, BG_B, BG_G, BG_R):
        for i in range(len(frame)):
            a = 1

    """Control Parameters"""
    UpBasedOnNowBG = 0
    UpWithoutNowBG = 1 - UpBasedOnNowBG
    """Imported Values"""
    [h, w] = frame[0].shape  # Get the shape
    [B, G, R] = frame[0:3]  # Get the Color Frame
    S = frame[3]  # Get the light strength map
    # Get the colored background[BG_B, BG_G, BG_R, BG_flag]
    # ,and a matrix BG_flag as the indicator of Background Place
    [BG_B, BG_G, BG_R, BG_flag] = BG
    [weight, miu, sigma, N, M, Z] = [PAR.weight, PAR.miu, PAR.sigma, PAR.N, PAR.M, PAR.Z]
    """Calculate P(L_t|I(x, y, t), @theta)"""
    P = weight / (np.sqrt(2 * math.pi) * sigma) * np.exp(
        -1 * (FrameInNdarray - miu) * (FrameInNdarray - miu) / 2 / (sigma * sigma))
    N = N + P
    M = M + P * FrameInNdarray
    Z = Z + P * FrameInNdarray * FrameInNdarray
    sumN = N
    sumN[:, :, :, 0] = np.sum(sumN, 3)
    sumN[:, :, :, 1] = np.sum(sumN, 3)
    sumN[:, :, :, 2] = np.sum(sumN, 3)
    weight = N / sumN
    miu = M / N
    sigma = Z / N - miu * miu

    """Update Part"""
    """
    for j in range(h):
        for i in range(w):
            if UpWithoutNowBG:
                UpDateWithoutNowBG(frame, BG_B, BG_G, BG_R)
            else:
                a = 0
    """


def InitialPars(frame, PAR):
    print("正在初始化全图参数")
    [h, w, chn, t] = frame.shape
    print([w, h, chn])
    for i in range(chn):
        sample = frame[:, :, i, :]
        PAR.miu[i, :, :, 0] = np.mean(sample, 2)
        PAR.sigma[i, :, :, 0] = np.var(sample, axis=2)
        PAR.M[i, :, :, :] = PAR.N[i, :, :, :] * PAR.miu[i, :, :, :]
        PAR.Z[i, :, :, :] = PAR.N[i, :, :, :] * (PAR.sigma[i, :, :, :] + PAR.miu[i, :, :, :] * PAR.miu[i, :, :, :])
    print("全图参数初始化完毕")


def GetSub(BG, frame):
    print("这里是获得背景差分结果的方程")
