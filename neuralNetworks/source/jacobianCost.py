import numpy as np
from feedForward import *
from inflateMatrix import *
from deflateMatrix import *

def jacobianCost(H, X, Y, shapes):
    #Setup
    H = inflateMatrix(H,shapes)
    m, layers = len(X), len(H) + 1
    a = feedForward(X, H)
    deltas = [a[-1] - Y] 
    #Back prop
    for i in range(layers - 1 , 0 , -1):
        deltas.insert(0, #In wich position it is inserted
            np.multiply(
                np.matmul(
                    deltas[0],
                    H[i-1][:,1:]
                ),
                np.multiply(
                    (a[i-1]), 
                    (1 - a[i-1])
                )
            )
        )

    derivate = []
    
    for i in range(layers - 1):
        derivate.append(
            np.matmul(
               deltas[i+1].T,
                np.hstack((
                    np.ones(len(X)).reshape(len(X), 1), #add bias
                    a[i]
                ))
            ) / m
        )
    return deflateMatrix(derivate)
