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
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import subprocess
import os
import seaborn as sns 
import itertools
from random import sample 

##for save model optimal
import pickle

from model import *
from productmodel import make_prection
from utildb import function_nettoyage
from stat_desc import *
"declaration des constances"
name_data_apprentissage="data/labeled_dataset.csv"
name_data_prediction="data/scoring_dataset.csv"

# la liste des modèles a tester 'linear_model_OLS',
model_name=['MLPregression','svm_m','SGDRegressor_m','BayesianRidge_m','SGDRegressor_m','RandomForestRegressor_m','treeDecision_regresion_m','KNeighborsRegressor_m','linear_model_OLS','AdaBoostRegressor_m']
#model_name=['AdaBoostRegressor_m']



"nbre de fois que l'algorithme devra tournée pour concerver le mielleur de tous pour un model precise"
nbre_fois=3
Y_name="Benefice net annuel"
tampon_value=dict()
#pourecntage des données spliter pour l'apprentissage et l'autre moitie pour les tests
perc=23 

seuil=0.90

if sys.platform!='win32': 
    cmd='mkdir '+os.getcwd()+'/model'
    os.system(cmd)
    cmd='mkdir '+os.getcwd()+'/simulation'
    os.system(cmd)
else:
    cmd='mkdir '+os.getcwd()+'\\model'
    os.system(cmd)
    cmd='mkdir '+os.getcwd()+'\\simulation'
    os.system(cmd)    
    

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

#statistique descriptives

"plot le graphe"
make_plot(train)

"plot la correlation"
make_correlation(train)

"faire les tests"
test_static,test_static1=make_test_statistics(X,Y)

"faire les ACP"
make_pca(X)


"faire la combinaision des varaible necessaire pour l'optimisation du choix des paramettre"
selection_varaible=[]
for p in itertools.chain(*(itertools.combinations(list(X.columns),long) for long in range(1,len(list(X.columns))))):
    selection_varaible.append(p)

# apprentissage du model
    
Score_max=dict()
#    



for g in range(0,4000):
    print("========"+str(g)+"===========\n")
    try:
        #memoire_score=dict()
        chemin_score1=None
        #list_variable=selection_varaible[len(selection_varaible)-1-g]
        if g!=0:
            list_variable=list(sample(selection_varaible,1)[0])
        else:
            list_variable=list(X.columns)
        
        run_model(X,Y,int(len(list(X.index))*perc/100),nbre_fois,model_name,list_variable)
        # prediction des données
        print("========je suis rentrée dans la prediction===========\n")
        try:
            chemin_score1=os.getcwd()+'/model_high'+str(g)+'.score'
            yprediction=make_prection(name_data_prediction,tampon_value,model_name,list_variable)
            for keys in list(memoire_score.keys()):
                if memoire_score[keys]>seuil:
                    if sys.platform!='win32':
                        chemin=os.getcwd()+'/simulation/'+keys+str(g)+'.csv'
                        chemin_score=os.getcwd()+'/simulation/model'+str(g)+'.score'
                        chemin_score1=os.getcwd()+'/model_high'+str(g)+'.score'
                        cmd='cp '+os.getcwd()+'/model/'+keys+'.pkl '+os.getcwd()+'/simulation/'+keys+str(g)+'.pkl'
                    else :
                        chemin=os.getcwd()+'\\simulation\\'+keys+str(g)+'.csv'
                        chemin_score=os.getcwd()+'\\simulation\\model'+str(g)+'.score'
                        chemin_score1=os.getcwd()+'\\model_high'+str(g)+'.score'                    
                        cmd='copy '+os.getcwd()+'\\model\\'+keys+'.pkl '+os.getcwd()+'\\simulation\\'+keys+str(g)+'.pkl'
                    os.system(cmd)
                    print(cmd)
                    #cette ligne sert a enregistrer les données 
                    pd.DataFrame(yprediction[keys]).to_csv (chemin, index = True, header=True)
                    f = open(chemin_score,"w")
                    f.write( str(memoire_score))
                    f.close()
                    print("========"+str(chemin)+"===========\n")
        except:
            pass
        Score_max[g]={"variable":list_variable,"score":memoire_score}
        f1 = open(chemin_score1,"w")
        f1.write( str(Score_max))
        f1.close()
        print("========filesave===========\n")
    except:
        pass
    print("========"+str(g)+"===========\n")
    print("========Next===========\n")
