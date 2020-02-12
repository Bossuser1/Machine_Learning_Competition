#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:48:33 2020

@author: traore
"""
import os
import pandas as pd 
import numpy as np
from sklearn.externals import joblib
from utildb import function_nettoyage
import sys
from sklearn import tree


def make_prection(name_data,tampon_value,model,list_variable):
    train= pd.read_csv(name_data)
    X=function_nettoyage(train,tampon_value)
    #######Make prediction ########"
    df_prec=pd.DataFrame() 
    dg=[]
    
    if sys.platform!='win32':
       chemin=os.getcwd()+'/model/'
    else:
       chemin=os.getcwd()+'\\model\\' 
    X=X[list(list_variable)]
    # Load the pickle file
    for name_model in model:
        clf_load = joblib.load(chemin+name_model+'.pkl')
        dg.append(clf_load.predict(X))
    
    return pd.DataFrame(np.transpose(dg),columns=model)  


