import numpy as np
from sigmoid import *

def feedForward(X, H):
    a = [X]
    for i in range(len(H)):
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((
                        np.ones(len(X)).reshape(len(X), 1),
                        a[i]
                    )) ,
                    H[i].T
                )
            )
        )
        
    return a