import numpy as np
def deflateMatrix(thetas): 
    return np.asarray(
        np.append(
            np.reshape(thetas[0],np.shape(thetas[0])[0]*np.shape(thetas[0])[1]),
            np.reshape(thetas[1],np.shape(thetas[1])[0]*np.shape(thetas[1])[1])
        )
    )
