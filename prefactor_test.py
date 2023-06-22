import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from tqdm import tqdm
n=2
steps=1+2*n
mat=np.zeros(steps)
mat[1+n]=1
for i in range(n):
    for j in range(n):
        newmat=[i,j]=mat[i-1,j-1] +mat[i-1, j+1]



print(mat)