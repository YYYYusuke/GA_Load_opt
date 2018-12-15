import numpy as np
from GA import myGA
import pickle
import sklearn
import scipy
from sklearn import preprocessing

Load_inputs=[4, 10, 23, 3, 7, 8, 36.2, 34.9, 35.1]

filename = 'sc_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X=loaded_model.transform([Load_inputs])

print(X)

filename = 'svr_rbf_model_KID11.sav'
loaded_model = pickle.load(open(filename, 'rb'))

Y_pred= loaded_model.predict(X)

print(Y_pred)