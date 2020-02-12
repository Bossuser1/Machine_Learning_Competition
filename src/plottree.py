#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:37:14 2020

@author: traore
"""
#!/usr/bin/env python
# coding: utf-8
from __future__ import (print_function, division,
    absolute_import, unicode_literals)
import sys
import warnings
import pandas as pd 
import numpy as np
import matplotlib
from sklearn import preprocessing
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import subprocess
import os
import seaborn as sns 
import itertools
from random import sample 
from sklearn.externals import joblib

##for save model optimal
import pickle
from model import *
from productmodel import make_prection
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
from sklearn.tree.export import export_text
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()

from utildb import function_nettoyage
from sklearn.metrics import mean_squared_error
name_data_apprentissage="data/labeled_dataset.csv"
name_data_prediction="data/scoring_dataset.csv"


width_in_inches = 10
height_in_inches =10
dots_per_inch =10
#
plt.figure(
   figsize=(width_in_inches, height_in_inches),
    dpi=dots_per_inch)


Y_name="Benefice net annuel"
tampon_value=dict()

#lecture des données 
train= pd.read_csv(name_data_apprentissage)


"valeur manquante corrigée"
if type(tampon_value)!=int:
    for name in list(train.columns):
        try:
            tampon_value[name]=train[name].mean()
        except:
            tampon_value[name]=list(train[name].describe())[2]
            pass

#creation de la vriable expliquéée
Y = train[Y_name]

#netoyyage du dataset X
X=function_nettoyage(train,tampon_value)


#tree.plot_tree(clf)
chemin=os.getcwd()+"/model0/"
name_model='treeDecision_regresion_m0'

clf_load = joblib.load(chemin+name_model+'.pkl')
score_tree=clf_load.score(X,Y)
yTest_predicted =clf_load.predict(X)
rmse=mean_squared_error(yTest_predicted,Y)

name_model='RandomForestRegressor_m0'

pd.DataFrame(yprediction[keys]).to_csv (chemin, index = True, header=True)

clf_load = joblib.load(chemin+name_model+'.pkl')
score_tree=clf_load.score(X,Y)
yTest_predicted =clf_load.predict(X)
rmse=mean_squared_error(yTest_predicted,Y)

pd.DataFrame(yprediction[keys]).to_csv (chemin, index = True, header=True)

name_model='AdaBoostRegressor_m0'

clf_load = joblib.load(chemin+name_model+'.pkl')
score_tree=clf_load.score(X,Y)
yTest_predicted =clf_load.predict(X)
rmse=mean_squared_error(yTest_predicted,Y)

pd.DataFrame(yprediction[keys]).to_csv (chemin, index = True, header=True)




## Create DOT data
#dot_data = tree.export_graphviz(clf_load, out_file=None)#, 
##                                #feature_names=iris.feature_names,  
##                                #class_names=iris.target_names)
#
## Draw graph
#graph = pydotplus.graph_from_dot_data(dot_data)  
#
#r = export_text(clf_load,list(X.columns))
#
#tree.plot_tree(clf_load,feature_names=list(X.columns))




# Show graph
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())
