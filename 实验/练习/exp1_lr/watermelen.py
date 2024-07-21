import pandas as pd
import numpy as np
from Logistic_Regression import Logistic_Regression as lr

file=open('exp1_lr\watermelon.csv',encoding='UTF-8')
data=np.array(pd.read_csv(file).values.tolist())
X=data[:,1:-1]
y=data[:,-1]
c1=1e-4
c2=0.9
times=int(10)
delta=1e-3


x=lr(X,y,c1,c2,times,delta)
w=x.Binary_Classification(0,0.1)
print(w)