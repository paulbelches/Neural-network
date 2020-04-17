import numpy as np
def inflateMatrix(flatThetas, shapes):
    n = shapes[0][0] * shapes[0][1]
    inflateMatrix = [flatThetas[1 : n + 1 ].reshape(shapes[0][0], shapes[0][1])]
    inflateMatrix.append(flatThetas[n : ].reshape(shapes[1][0], shapes[1][1]))
    return inflateMatrix