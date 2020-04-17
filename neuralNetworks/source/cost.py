import numpy as np
from feedForward import *
from inflateMatrix import *

def cost(H, X, Y, shapes):
    a = feedForward(X, inflateMatrix(H,shapes))
    return -( Y * np.log(a[-1]) + (1 - Y) * np.log(1- a[-1]) ).sum() / len(X)