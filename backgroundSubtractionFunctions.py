import numpy as np


class PixelPar:
    def __init__(self, size):
        self.weight = np.zeros(size)
        self.miu = np.zeros(shape=size, dtype=np.uint8)
        self.sigma = np.zeros(size)
        self.nGM = np.ones(shape=size, dtype=np.uint8)


def updateBackGround(background, frame, PAR):
    [w, h] = frame.shape



def InitialPars(frame, PAR):
    [h, w, chn, t] = frame.shape
    print([w, h, chn])
    sample = frame[:, :, 1, :]
    PAR.miu[:, :, 1] = np.mean(sample, 2)
    PAR.sigma[:, :, 1] = np.var(sample, axis=2)
    PAR.weight = np.ones((h, w, chn))
    """
    for i in range(w):
        for j in range(h):
            weight = 1
            PAR.weight[j, i, 1] = weight
    """
