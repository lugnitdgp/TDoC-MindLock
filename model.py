import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('./sudoku.csv')
X=np.array(df.quizzes.map(lambda x: list(map(int, x))).to_list())
Y=np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())
X=X.reshape(-1,9,9,1)
Y=Y.reshape(-1,9,9)-1
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
