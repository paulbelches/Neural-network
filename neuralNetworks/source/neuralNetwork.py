from jacobianCost import *
from cost import *
import scipy.optimize as op
import numpy as np
shapes = []
print("Welcome")
#Read Data
print("Start reading csv..")
data = np.genfromtxt('mnist-in-csv/mnist_train.csv', delimiter=',')
data = np.delete(data, 0, 0)
#Slide into X and R
print("Finish reading csv")
X = data[:,1:]
#print(X)
X = X / 1000
rowNumber = np.size(X,0)
print("Generating Y..")
Y = np.zeros( rowNumber * 10).reshape(rowNumber, 10)
result= data[:,0]
for i in range(len(result)):
    Y[i][int(result[i])] = 1
print("Finish Generating Y")
#Generate Theta
print("Generating Theta")
neuronalNumber = np.shape(X)[1] //2
#Shape of cost of transition into hidden layer
shapes.append( (neuronalNumber, neuronalNumber*2 + 1) )
#H = [np.random.rand(neuronalNumber, neuronalNumber*2 + 1) / 10000] 
#Shape of cost of transition out hidden layer
shapes.append( (10, neuronalNumber + 1) )
np.random.seed(seed=100)
H = np.random.rand( (shapes[0][0]*shapes[0][1]) + (shapes[1][0]*shapes[1][1]) )
H = np.asarray(H) 
print("Finish generating Theta")
#Feed forward
#print(np.shape(H))
#for i in shapes:
#    print(i)

#for i in jacobianCost(H, X, Y, shapes):
#    print( np.shape(i) )
jacobianCost(H, X, Y, shapes)
#print( np.shape(jacobianCost(H, X, Y, shapes)) ) 
'''
result = op.minimize(
    fun=cost,
    x0=H,
    args=(X, Y, shapes),
    method='L-BFGS-B',
    jac=jacobianCost,
    options={'disp':True, 'maxiter':500}
)
print('##############################')
print(np.shape(result.x))
print(np.shape(H))
print(result)
'''