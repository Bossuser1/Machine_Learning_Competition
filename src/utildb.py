#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:35:30 2020

@author: traore
"""
import os
import pandas as pd 
import numpy as np
import sys

# fucntion utiliser dans la predction mais aussi dans l'apprentissage 
def function_nettoyage(train, tampon_value):
    " cette fonction sert a recoder les données , pour les missings values , j'ai utiliser une valeure tampon"
    X_name=["Age","Prime mensuelle","Categorie socio professionnelle","Kilometres parcourus par mois","Coefficient bonus malus","Type de vehicule","Score CRM","Niveau de vie","Marque","Salaire annuel","Score credit","Cout entretien annuel"]
    X= train[X_name]
       
    if type(tampon_value)!=int:
        # recodage des NA
        for name_variable in  list(X.columns):
            if name_variable not in ["Categorie socio professionnelle","Marque","Type de vehicule"]:
                try:
                    X[name_variable]=X[name_variable].fillna(tampon_value[name_variable]) #remplacer les na par la mediane 
                except:
                    pass
            else:
                try:
                    X[name_variable]=X[name_variable].fillna(tampon_value[name_variable]) #remplacer les na par la mediane 
                except:
                    pass           
    else:
        # recodage des NA
        for name_variable in  list(X.columns):
            if name_variable not in ["Categorie socio professionnelle","Marque","Type de vehicule"]:
                try:
                    X[name_variable]=X[name_variable].fillna(tampon_value) #remplacer les na par la mediane 
                except:
                    pass
            else:
                try:
                    X[name_variable]=X[name_variable].fillna(tampon_value) #remplacer les na par la mediane 
                except:
                    pass    
    if type(tampon_value)!=int:
        dict2 = {"Peugeot": 1, "Renault": 2, "Toyota": 3, "Opel": 4,"Citroen":5,"Volkswagen":6,np.nan:tampon_value['Marque']}
        dict3 = {"Etudiant": 1, "Ouvrier": 2, "Cadre": 3, "Sans emploi": 4,"Travailleur non salarie":5,np.nan:tampon_value['Categorie socio professionnelle']}
        dict1 = {"SUV": 1, "5 portes": 2, "3 portes": 3, "Utilitaire": 4,np.nan:tampon_value['Type de vehicule']}
    else:
        dict2 = {"Peugeot": 1, "Renault": 2, "Toyota": 3, "Opel": 4,"Citroen":5,"Volkswagen":6,np.nan:tampon_value}
        dict3 = {"Etudiant": 1, "Ouvrier": 2, "Cadre": 3, "Sans emploi": 4,"Travailleur non salarie":5,np.nan:tampon_value}
        dict1 = {"SUV": 1, "5 portes": 2, "3 portes": 3, "Utilitaire": 4,np.nan:tampon_value}
        
    
    #avoir les valeurs uniques des variables 
    X['Marque'].unique()
    # recodage des variables alphanumeriques 
    #X = X.replace({"Marque": dict2})
    X["Marque"]=X["Marque"].map(dict2)
    
    #Cas categorie sociodemographique
    X['Categorie socio professionnelle'].unique()
    # recodage des variables alphanumeriques 
    #X = X.replace({"Categorie socio professionnelle": dict3})
    X["Categorie socio professionnelle"]=X["Categorie socio professionnelle"].map(dict3)
    #cas du type de vehicule
    X['Type de vehicule'].unique() 
    
    #X = X.replace({"Type de vehicule": dict1}) 
    X["Type de vehicule"]=X["Type de vehicule"].map(dict1)


    return X

#
#
## In[ ]:
#
#
#
#
#
## In[103]:
#
#
#X.isnull().sum()
#
#
## In[104]:
#
#
## avoir les pourcentages de variables manquantes pour X
#X.isnull().sum()/len(X)
#
#
## In[ ]:
#
#
#
#
#
## In[105]:
#
#
#
## In[106]:
#
#
## avoir les pourcentages de variables manquantes pour Y
#Y.isnull().sum()/len(Y)
#
## ### recodage des varaibles 
#
#
### Ici on affiche bien que toutes les variables sont avec un effectif de 1000 
##contrairement a l'étape précédentes ou il y avait des NA
#X.describe() 
#
#
## In[109]:
#
#
##name="Age"
##X[name]
##np.transpose(pd.DataFrame(X.groupby([name])[name].count())) # 
##global median_age
### isoler les bonnes données des mauvaises pour la variable age
##filter1 = X[name]>17.0
##filter2 = X[name]<124.0
##median_age=X[name][filter1 & filter2].median()
##
##def f1(x):
##    if x<18.0:
##        return median_age
##    elif x>87.0:
##        return median_age
##    else:
##        return x
##X[name]= X[name].map(f1)
##np.transpose(pd.DataFrame(X.groupby([name])[name].count()))# age corrigé
##
## In[110]:
#name="Prime mensuelle"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) #  poser des questions sur les primes mensuelles ??
#
#
## In[111]:
#
#
#name="Kilometres parcourus par mois"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count()))
#
#
## In[112]:
#
#
#name="Score CRM"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le score CRM ??
#
#
## In[113]:
#
#
#name="Coefficient bonus malus"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le coefficient du bonus malus ??
#
#
## In[114]:
#
#
#name="Score CRM"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier le score CRM ??
#
#
## In[115]:
#
#
#name="Niveau de vie"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### verifier Niveau de vie ?? 
#
#
## In[116]:
#
#
#name="Marque"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) 
#
#
## In[117]:
#
#
#name="Salaire annuel"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### bizarre comme salaire ??
#
#
## In[118]:
#
#
#name="Score credit"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) ### veriifier si c est exact que ca commence par UN  ??
#
#
## In[119]:
#
#
#name="Cout entretien annuel"
#X[name]
#np.transpose(pd.DataFrame(X.groupby([name])[name].count())) 
#
#
## In[120]:
#
#
#X_Y=X.merge(Y,left_index=True, right_index=True)
#for name in list(X_Y.columns):
#    for name2 in list(X_Y.columns):
#        if name2!=name:
#            # ici on fait les test statistiques et on comparer a 0.00cinq comme seuil mais on peut reduire encore
#            print("variable1 :"+name+" variable2 :"+name2)
#            print(stats.kruskal(X_Y[name],X_Y[name2]).pvalue)
##### l'erreur que font plusieurs personnes est de ne pas tenir compte du fait qu'il s'agit d'un test multiple donc on doit utiliser 
##### un correcteur de bonferroni.
##### important a utiliser ici pour gagner d'enorme point  !!!!!!!!!!!!!!!!!!!!!!!!
#
#
## In[64]:
#
#
#plt.figure(figsize=(12,10))
#cor = X_Y.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()
#
#
## In[65]:
#
#
##### modelisation 
#
#
## In[123]:
#
#
#### Reseaux des neuronnés
#
#
## In[146]:
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4)
##train_test_split(X,y)

