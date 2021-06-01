import numpy as np

def numericalDerivative(f, x):
    deltaX = 1e-4 
    coefficient = np.zeros_like(x)    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        
        tmp_val = x[idx]

        x[idx] = tmp_val + deltaX     
        fx1 = f(x) # f(x+deltaX)
                
        x[idx] = tmp_val - deltaX 
        fx2 = f(x)   # f(x-deltaX)

        coefficient[idx] = (fx1 - fx2) / (2*deltaX)
                
        x[idx] = tmp_val
        it.iternext()
        
    return coefficient

def func1(input_obj): 
    w = input_obj[0, 0]
    x = input_obj[0, 1] 
    y = input_obj[1, 0]
    z = input_obj[1, 1] 
    return ( w*x + x*y*z + 3*w + z*np.power(y,2) )
 
input = np.array([ [1.0, 2.0], [3.0, 4.0] ])
result=numericalDerivative( func1, input )
print(result)
