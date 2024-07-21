import pandas as pd
from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.datasets import load_linnerud,load_diabetes

data=load_iris(as_frame=True)
dataset=data.frame
dataset.to_csv("iris.txt")

data=load_wine(as_frame=True)
dataset=data.frame
dataset.to_csv("wine.txt")

# data=load_digits(as_frame=True)
# dataset=data.frame
# dataset.to_csv("digits.txt")

data=load_linnerud(as_frame=True)
dataset=data.frame
dataset.to_csv("linnerud.txt")

data=load_diabetes(as_frame=True)
dataset=data.frame
dataset.to_csv("diabetes.txt")

