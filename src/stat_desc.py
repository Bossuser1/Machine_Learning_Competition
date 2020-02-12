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
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
##for save model optimal
import pickle
from scipy.stats import ttest_ind, f_oneway
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib import pyplot as plt
import seaborn as sns

def plot_graph(train,var1,var2,var3,type1,type2,type3):
    plt.rcParams["figure.figsize"] = (20,3)
    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=False)
    #ax1.axis('equal')
    #var1='Categorie socio professionnelle'
    ax1.set_title(var1)
    lab1 = list(pd.DataFrame(train[var1].value_counts()).index)
    val1 = list(pd.DataFrame(train[var1].value_counts())[var1])
    #ax1.pie(val1, labels = lab1,autopct='%1.2f%%')
    if type1==1:
        ax1.pie(val1, labels = lab1,autopct='%1.2f%%')
    elif type1==2:
        
        ax1.bar(lab1, val1, color='b')
    else:
        ax1.boxplot(val1)

    #var2='Marque'
    #ax2.axis('equal')
    ax2.set_title(var2)
    lab2 = list(pd.DataFrame(train[var2].value_counts()).index)
    val2 = list(pd.DataFrame(train[var2].value_counts())[var2])

    if type2==1:
        ax2.pie(val2, labels = lab2,autopct='%1.2f%%')
    elif type2==2:
        #width=10, width
        #np.arange(len(lab2))
        ax2.bar(lab2, val2, color='b')
        #ax2.bar(val2,lab2) #
    else:
        ax2.boxplot(val2)

    #var3='Type de vehicule'
    lab3 = list(pd.DataFrame(train[var3].value_counts()).index)
    val3 = list(pd.DataFrame(train[var3].value_counts())[var3])
    #ax3.bar(lab3,val3)
    #ax3.axis('equal')
    ax3.set_title(var3)

    if type3==1:
        ax3.pie(val3, labels = lab3,autopct='%1.2f%%')
    elif type3==2:
        ax3.bar(lab3, val3, color='b')
    else:
        ax3.boxplot(val3)
    plt.show()


def make_plot(train):
    "faire les graphes pour la representation"
    plot_graph(train,'Age','Prime mensuelle','Categorie socio professionnelle',3,3,1)
    
    plot_graph(train,'Kilometres parcourus par mois','Coefficient bonus malus','Type de vehicule',3,3,1)
    
    plot_graph(train,'Score CRM', 'Niveau de vie', 'Marque',3,3,1)
    
    
    plot_graph(train,'Salaire annuel', 'Score credit', 'Cout entretien annuel',3,3,3)
    
    plot_graph(train,'Categorie socio professionnelle','Marque', 'Benefice net annuel',2,1,3)



def make_correlation(train):
    # coefficient de correlation lineaire
    plt.figure(figsize=(12,10))
    cor = train[list(train.columns)[1:]].corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.show()


# tests statistics 
#stastics descriptives # tests statistics selections des variables
def make_test_statistics(X,Y):
    "faire les tests staistics et corrigées"
    test_static=dict()
    test_static1=dict()
    X_Y=X.merge(Y,left_index=True, right_index=True)
    for name in list(X_Y.columns):
        for name2 in list(X_Y.columns):
            if name2!=name:
                # ici on fait les test statistiques et on comparer a 0.00cinq comme seuil mais on peut reduire encore
                print("variable1 :"+name+" variable2 :"+name2)
                test_static[str(name)+'_'+str(name2)]=stats.kruskal(X_Y[name],X_Y[name2]).pvalue
                #print(stats.kruskal(X_Y[name],X_Y[name2]).pvalue)
    #une correction dû aux tests multiples
    bh=multipletests(list(test_static.values()), method = 'fdr_bh')
    
    for key in range(len(test_static.keys())):
        test_static1[list(test_static.keys())[key]]=bh[1][key]
    return test_static,test_static1



def make_pca(X):
    # Pca
    #pour la reduction des varaibles
    from sklearn.decomposition import PCA
    
    # PCA with scikit-learn
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    PC = pca.transform(X)
    

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    plt.scatter(PC[:, 0], PC[:, 1])
    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    
    # PCA with scikit-learn
    pca = PCA(n_components=3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    PC = pca.transform(X)
    
    #plt.scatter(PC[:, 0], PC[:, 1])
    #plt.subplot(122)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(PC[:, 0], PC[:, 1], PC[:, 2])
    ax.view_init(1, -10)
    #plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    #plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])

# ==============================================
#
# Part 4: Multiple Tests and P-Value Correction
#
# ==============================================

#### selection de varaibles
