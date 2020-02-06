#!/usr/bin/env python
# coding: utf-8

# In[96]:
# lecture de la base 
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import svm,tree ### pour les SVM
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error # RMSE
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
##for save model optimal
from keras.models import model_from_json
import pickle

from model import *

from random import sample 




get_ipython().run_line_magic('matplotlib', 'inline')

train= pd.read_csv("data/labeled_dataset_axaggpdsc.csv")


# In[97]:


train


# In[98]:


train.head(10)


# In[99]:


train.info()


# In[100]:


train.shape


# In[101]:


#creation de la vriable expliquéée
Y_name="Benefice net annuel"
Y = train[Y_name]

#Création des variables explicatives
X_name=["Age","Prime mensuelle","Categorie socio professionnelle","Kilometres parcourus par mois","Coefficient bonus malus","Type de vehicule","Score CRM","Niveau de vie","Marque","Salaire annuel","Score credit","Cout entretien annuel"]
X= train[X_name]


# In[102]:


X


# In[ ]:





# In[103]:


X.isnull().sum()


# In[104]:


# avoir les pourcentages de variables manquantes pour X
X.isnull().sum()/len(X)


# In[ ]:





# In[105]:


# recodage des NA
for name_variable in  list(X.columns):
    if name_variable not in ["Categorie socio professionnelle","Marque","Type de vehicule"]:
        try:
            X[name_variable]=X[name_variable].fillna(X[name_variable].median()) #remplacer les na par la mediane 
        except:
            pass
    else:
        try:
            X[name_variable]=X[name_variable].fillna(list(X[name_variable].describe())[2]) #remplacer les na par la mediane 
        except:
            pass        
X.isnull().sum()/len(X)


# In[106]:


# avoir les pourcentages de variables manquantes pour Y
Y.isnull().sum()/len(Y)


# ### recodage des varaibles 

# In[107]:


#avoir les valeurs uniques des variables 
X['Marque'].unique()
# recodage des variables alphanumeriques 
dict2 = {"Peugeot": 1, "Renault": 2, "Toyota": 3, "Opel": 4,"Citroen":5,"Volkswagen":6}
X = X.replace({"Marque": dict2})

#Cas categorie sociodemographique
X['Categorie socio professionnelle'].unique()
# recodage des variables alphanumeriques 
dict3 = {"Etudiant": 1, "Ouvrier": 2, "Cadre": 3, "Sans emploi": 4,"Travailleur non salarie":5}
X = X.replace({"Categorie socio professionnelle": dict3})

#cas du type de vehicule
X['Type de vehicule'].unique() 

dict1 = {"SUV": 1, "5 portes": 2, "3 portes": 3, "Utilitaire": 4}
X = X.replace({"Type de vehicule": dict1})
X


# In[108]:


## Ici on affiche bien que toutes les variables sont avec un effectif de 1000 
#contrairement a l'étape précédentes ou il y avait des NA
X.describe() 


# In[109]:


name="Age"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) # 
global median_age
# isoler les bonnes données des mauvaises pour la variable age
filter1 = X[name]>17.0
filter2 = X[name]<124.0
median_age=X[name][filter1 & filter2].median()

def f1(x):
    if x<18.0:
        return median_age
    elif x>87.0:
        return median_age
    else:
        return x
X[name]= X[name].map(f1)
np.transpose(pd.DataFrame(X.groupby([name])[name].count()))# age corrigé


# In[110]:


name="Prime mensuelle"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) #  poser des questions sur les primes mensuelles ??


# In[111]:


name="Kilometres parcourus par mois"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count()))


# In[112]:


name="Score CRM"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le score CRM ??


# In[113]:


name="Coefficient bonus malus"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le coefficient du bonus malus ??


# In[114]:


name="Score CRM"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le score CRM ??


# In[115]:


name="Niveau de vie"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier Niveau de vie ?? 


# In[116]:


name="Marque"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) 


# In[117]:


name="Salaire annuel"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### bizarre comme salaire ??


# In[118]:


name="Score credit"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### veriifier si c est exact que ca commence par UN  ??


# In[119]:


name="Cout entretien annuel"
X[name]
np.transpose(pd.DataFrame(X.groupby([name])[name].count())) 


# In[120]:


X_Y=X.merge(Y,left_index=True, right_index=True)
for name in list(X_Y.columns):
    for name2 in list(X_Y.columns):
        if name2!=name:
            # ici on fait les test statistiques et on comparer a 0.00cinq comme seuil mais on peut reduire encore
            print("variable1 :"+name+" variable2 :"+name2)
            print(stats.kruskal(X_Y[name],X_Y[name2]).pvalue)
#### l'erreur que font plusieurs personnes est de ne pas tenir compte du fait qu'il s'agit d'un test multiple donc on doit utiliser 
#### un correcteur de bonferroni.
#### important a utiliser ici pour gagner d'enorme point  !!!!!!!!!!!!!!!!!!!!!!!!


# In[64]:


plt.figure(figsize=(12,10))
cor = X_Y.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[65]:


#### modelisation 


# In[123]:


### Reseaux des neuronnés


# In[146]:
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4)
#train_test_split(X,y)
perc =int(len(list(X.index))*40/100) 



def calcul_gini(actual_list, predictions_list):
    try:
        fpr, tpr, thresholds = roc_curve(actual_list, predictions_list)
        roc_auc = auc(fpr, tpr)
        GINI = (2 * roc_auc) - 1
        return GINI
    except:
        return "False"
        pass


X=(X-X.mean())/X.std()
Y=(Y-Y.mean())/Y.std()

for k in range(1,10):
    train_index=[]
    test_index=sample(list(X.index),perc)
    for k in list(X.index):
        if k not in test_index:
            train_index.append(k)
    X_test=X.iloc[test_index] # Chargement du X test
    y_test=Y.iloc[test_index] # Chargement du y test

    X_train=X.iloc[train_index] # Chargement du X train
    y_train=Y.iloc[train_index] # Chargement du y train    
    
    "model lineaire"
    #try:
    #    "Ordinary Least Squares"
    #    linear_model_OLS(X_train, y_train,X_test, y_test)
    #except:
    #    pass
    
    #for alph in [0.1,0.4,0.5,0.6,0.7,0.8]:
    #    try:
    #        "Ordinary Least Rigde"
    #        linear_model_Ridge(X_train, y_train,X_test, y_test,alph)
    #    except:
    #        pass
    #for statut in [True,False]:
    #    try:
    #       BayesianRidge_m(X_train, y_train,X_test, y_test,statut)
    #    except:
    #        pass
    
    #for statut in ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"]:
    #    try:
    #       SGDRegressor_m(X_train, y_train,X_test, y_test,statut)
    #    except:
    #        pass
    
    "discrimation"
    #svm section regression
    #for statut in ["linear"]:
    #    for ty in ["LinearSVR","svm"]:
    #        try:
    #           svm_m(X_train, y_train,X_test, y_test,statut,ty)
    #        except:
    #            pass
            
    #gaussian_process_m
    #try:
    #    gaussian_process_m(X_train, y_train,X_test, y_test)
    #except:
    #    pass    
 
    
    # arbre de decision
    for maxdep in range(1,20):
        try:
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,maxdep)
        except:
            pass

    # arbre de decision
    for maxdep in range(1,20):
        try:
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,maxdep)
        except:
            pass



   
#        clf_SVR = svm.SVR()
#        clf_SVR.fit(X_train, y_train)
#        score_SVR = clf_SVR.score(X_test, y_test)
#        print(" SVM")
#        print(score_SVR)
#        y_pred=clf_SVR.predict(X_test)
#        print("gini:"+str(calcul_gini(y_test,y_pred)))
#        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
#    except:
#        pass    
#    try:
#        clf_tree = tree.DecisionTreeRegressor()
#        clf_tree.fit(X_train, y_train)
#        score_tree = clf_tree.score(X_test, y_test)
#        print(" arbre de decision")
#        print(score_tree)
#        y_pred=clf_tree.predict(X_test)
#        print("gini:"+str(calcul_gini(y_test,y_pred)))
#        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
#    except:
#        pass      
#    
#    "Neural network models (supervised)"
#    
#    for  solver in ["lbfgs","sgd","adam"]:
#        for activation in ["identity","logistic","tanh","relu"]:
#            for size_hidden in [(100,10) ,(70,30),(20,1)]:
#                try:
#                    MLPregression(X_train, y_train,X_test, y_test,solver,size_hidden,activation)
#                except:
#                    pass 
#
    
    
#    try:
#        # version classification
#        modelMLPC = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(10, 5))
#        #to learn
#        modelMLPC.fit(X_train, y_train)
#        #to predict
#        print("Neural network ")
#        yTest_predicted =modelMLPC.predict(X_test)
#        print(accuracy_score(y_test,yTest_predicted))
#    except:
#        pass    


#modelMLPC = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(10, 5))
#to learn
#modelMLPC.fit(X_train, y_train)
#to predict


#yTest_predicted =modelMLPC.predict(X_test)
#print(accuracy_score(y_test,yTest_predicted))
    #.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)



    #clf_SGDCl=SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    #clf_SGDCl.fit(X_train, y_train)
    #score_SGDCl = clf_SGDCl.score(X_test, y_test)
    #print(" arbre de decision")
    #print(score_SGDCl)    
    #y_pred=clf_SGDCl.predict(X_test)
    #print("gini:"+str(calcul_gini(y_test,y_pred)))
    #print("RMSE:"+str(mean_squared_error(y_test, y_pred)))

#def neural_network(XTrain, yTrain,XTest,yTest):    
#    model = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(10, 5))
#    #to learn
#    model.fit(XTrain, yTrain)
#    #to predict
#    yTest_predicted =model.predict(XTest)
#    accura=accuracy_score(yTest,yTest_predicted)
#    print("accuracy :"+str(accura)) 
#    
#    return accura,model
    

#    print("accuracy :"+str(accura)) 

