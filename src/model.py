#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:22:56 2020

@author: traore
"""
from random import sample 
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import mean_squared_error # RMSE
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
#from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.naive_bayes import CategoricalNB
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import os
import sys

##
global memoire_score

memoire_score={}


#def savemodel(func):
#    "mon decorateur "
#    def wrapper():
#        print("<!_____________!>")
#        func()
#        print("<!_____________!>")
#    return func
#    
def savemodel(func_to_decorate):
    def new_func(*original_args, **original_kwargs):
        print("Function has been decorated.  Congratulations."+str(func_to_decorate))
        # Do whatever else you want here
        #memoire_score[str(func_to_decorate)]=func_to_decorate(*original_args, **original_kwargs)
        return func_to_decorate(*original_args, **original_kwargs)
    return new_func



def sauvergarde_file_optimal(clf_tree,score_tree,name_model):
        " Call une varaible global pour voir ce qui était dans sa memoire comme modèle optimal"
        try:
           if sys.platform!='win32':
              chemin=os.getcwd()+'/model/'
           else:
              chemin=os.getcwd()+'\\model\\' 
            
           if score_tree>=memoire_score[name_model]:
              joblib.dump(clf_tree,chemin+name_model+'.pkl')
              memoire_score[name_model]=score_tree
           else:
               print("Not Max:")
        except:
            memoire_score[name_model]=score_tree
            joblib.dump(clf_tree, chemin+name_model+'.pkl')
            pass
        #print(os.getcwd()+'/model/'+name_model+'.pkl')




#Supervised learning


def linear_model_OLS(X_train, y_train,X_test, y_test,reg):
    "Ordinary Least Squares"
    try:
        reg.fit(X_train, y_train)
        print("Linear model Ordinary Least Squares")
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'linear_model_OLS')
        #sauvergarde_file_optimal(reg,score,'linear_model_OLS')
    except:
        pass



def linear_model_Ridge(X_train, y_train,X_test, y_test,alph,reg):
    "Ridge"
    try:
        reg.fit(X_train, y_train)
        print("Linear model Ridge"+str(alph))
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'linear_model_Ridge')
    except:
        pass

def BayesianRidge_m(X_train, y_train,X_test, y_test,reg):
    "Ridge"
    try:
        #reg=linear_model.BayesianRidge(compute_score=statut)
        reg.fit(X_train, y_train)
        print("BayesianRidge_m ")
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'BayesianRidge_m')
    except:
        pass

def SGDRegressor_m(X_train, y_train,X_test, y_test,reg):
    "SGDRegressor"
    try:
        #reg=linear_model.SGDRegressor(loss=statut,max_iter=1000, tol=0.001)
        reg.fit(X_train, y_train)
        print("SGDRegressor_m")
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'SGDRegressor_m')
    except:
        pass


def svm_m(X_train, y_train,X_test, y_test,clf_SVR):
        "svm"
        try:
            #if ty=="svm":
            #    clf_SVR = svm.SVR()
            #elif ty=="LinearSVR":
            #    clf_SVR = svm.LinearSVR()
            #else:
            #    pass
            clf_SVR.fit(X_train, y_train)
            score_SVR = clf_SVR.score(X_test, y_test)
            print(" SVM ")
            print(score_SVR)
            y_pred=clf_SVR.predict(X_test)
            print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
            "sauvegarder le modele"
            sauvergarde_file_optimal(clf_SVR,score_SVR,'svm_m')
        except:
            pass     


def gaussian_process_m(X_train, y_train,X_test, y_test,gpr):
        "gaussian_process"
        try:
            gpr.fit(X_train, y_train)
            score_gpr = gpr.score(X_test, y_test)
            print(score_gpr)
            y_pred=gpr.predict(X_test)
            print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
            "sauvegarder le modele"
            sauvergarde_file_optimal(gpr,score_gpr,'gaussian_process_m')
        except:
            pass        


#def naiveBaye_m(X_train, y_train,X_test, y_test):
#    "naaive bayes"
  
def treeDecision_regresion_m(X_train, y_train,X_test, y_test,clf_tree):
    #tree Descion
    score_tree=0
    try:
        #"regtreeDecision_regresion_m"+str(maxdep)+"=tree.DecisionTreeRegressor(max_depth="+str(maxdep)+")"
        clf_tree.fit(X_train, y_train)
        score_tree = clf_tree.score(X_test, y_test)
        print(" arbre de decision")
        print(score_tree)
        y_pred=clf_tree.predict(X_test)
        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(clf_tree,score_tree,'treeDecision_regresion_m')
    except:
        pass
    return  score_tree    

 
# randomForestRegression         
def RandomForestRegressor_m(X_train, y_train,X_test, y_test,clf_tree):
    #tree Descion
    score_tree=0
    try:
        #clf_tree = RandomForestRegressor(max_depth=maxt, random_state=0)
        clf_tree.fit(X_train, y_train)
        score_tree = clf_tree.score(X_test, y_test)
        print(" random forest ")
        print(score_tree)
        y_pred=clf_tree.predict(X_test)
        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(clf_tree,score_tree,'RandomForestRegressor_m')
    except:
        pass
    return score_tree

      
def MLPregression(X_train, y_train,X_test, y_test,modelMLPR):
    try:
        # version classification
        #if solver in ["lbfgs","sgd","adam"]:
        #    modelMLPR = MLPRegressor(activation=activation1,solver=solver, alpha=1e-7,hidden_layer_sizes=size_hidden,)
        #else:
        #    solver='lbfgs'
        #    modelMLPR = MLPRegressor(activation=activation1,solver=solver, alpha=1e-7,hidden_layer_sizes=size_hidden)
        
        #to learn
        modelMLPR.fit(X_train, y_train)
        #to predict
        print("Neural network ")
        yTest_predicted =modelMLPR.predict(X_test)
        score=modelMLPR.score(X_test, y_test)
        print("Score:"+str(score))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        sauvergarde_file_optimal(modelMLPR,score,'MLPregression')
    except:
        pass


def KNeighborsRegressor_m(X_train, y_train,X_test, y_test,reg):
    try:
        reg.fit(X_train, y_train)
        print("KNeighborsRegressor_m")
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'KNeighborsRegressor_m')
    except:
        pass
    
def AdaBoostRegressor_m(X_train, y_train,X_test, y_test,reg):
    try:
        #reg 
        reg.fit(X_train, y_train)
        print("AdaBoostRegressor_m")
        yTest_predicted =reg.predict(X_test)
        score=reg.score(X_test, y_test)
        print(score)
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
        "sauvegarder le modele"
        sauvergarde_file_optimal(reg,score,'AdaBoostRegressor_m')
    except:
        pass    
    
    
def calcul_gini(actual_list, predictions_list):
    try:
        fpr, tpr, thresholds = roc_curve(actual_list, predictions_list)
        roc_auc = auc(fpr, tpr)
        GINI = (2 * roc_auc) - 1
        return GINI
    except:
        return "False"
        pass 
    

def run_model(X,Y,perc,nbre_fois,model_name,list_variable):
    regRandomForestRegressor_m1,regRandomForestRegressor_m2,regRandomForestRegressor_m3,regRandomForestRegressor_m4,regRandomForestRegressor_m5=None,None,None,None,None
    "ensemble des modeles crées"
    memoire_score=dict()
    if 'RandomForestRegressor_m' in model_name:
        regRandomForestRegressor_m1=RandomForestRegressor(max_depth=2, random_state=0)
        regRandomForestRegressor_m2=RandomForestRegressor(max_depth=7, random_state=0)
        regRandomForestRegressor_m3=RandomForestRegressor(max_depth=12, random_state=0)
        regRandomForestRegressor_m4=RandomForestRegressor(max_depth=14, random_state=0)
        regRandomForestRegressor_m5=RandomForestRegressor(max_depth=18, random_state=0)

    if 'treeDecision_regresion_m' in model_name:
        regtreeDecision_regresion_m1=tree.DecisionTreeRegressor(max_depth=2)
        regtreeDecision_regresion_m2=tree.DecisionTreeRegressor(max_depth=7)
        regtreeDecision_regresion_m3=tree.DecisionTreeRegressor(max_depth=12)
        regtreeDecision_regresion_m4=tree.DecisionTreeRegressor(max_depth=14)
        regtreeDecision_regresion_m5=tree.DecisionTreeRegressor(max_depth=18)

    if 'svm_m' in model_name:
        regvm_m1=svm.SVR()
        regvm_m2=svm.LinearSVR()                
    
    if 'linear_model_OLS' in model_name:
        reglinearie = linear_model.LinearRegression()
    if 'KNeighborsRegressor_m' in model_name:
        regKNeighborsRegressor=KNeighborsRegressor(n_neighbors=2)
    
    if 'linear_model_Ridge' in model_name:
        reglinear_model_Ridge1 = linear_model.Ridge(alpha=0.1)
        reglinear_model_Ridge2 = linear_model.Ridge(alpha=0.1)
        reglinear_model_Ridge3 = linear_model.Ridge(alpha=0.1)
        reglinear_model_Ridge4 = linear_model.Ridge(alpha=0.1)
        reglinear_model_Ridge5 = linear_model.Ridge(alpha=0.5)
        reglinear_model_Ridge6 = linear_model.Ridge(alpha=0.6)
        reglinear_model_Ridge7 = linear_model.Ridge(alpha=0.7)
        reglinear_model_Ridge8 = linear_model.Ridge(alpha=0.8)
        reglinear_model_Ridge9 = linear_model.Ridge(alpha=0.9)
    if 'BayesianRidge_m' in model_name:
        regBayesianRidge_m_F=linear_model.BayesianRidge(compute_score=False)
        regBayesianRidge_m_T=linear_model.BayesianRidge(compute_score=True)
    if 'AdaBoostRegressor_m' in model_name:
        regAdaBoostRegressor_m=AdaBoostRegressor(random_state=0, n_estimators=100)
    if 'MLPregression' in model_name:
        reqMLPregression1=MLPRegressor(activation="identity",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression2=MLPRegressor(activation="identity",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression3=MLPRegressor(activation="identity",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression4=MLPRegressor(activation="logistic",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression5=MLPRegressor(activation="logistic",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression6=MLPRegressor(activation="logistic",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression7=MLPRegressor(activation="tanh",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression8=MLPRegressor(activation="tanh",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression9=MLPRegressor(activation="tanh",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression10=MLPRegressor(activation="relu",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression11=MLPRegressor(activation="relu",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression12=MLPRegressor(activation="relu",solver="lbfgs", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression13=MLPRegressor(activation="identity",solver="sgd", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression14=MLPRegressor(activation="identity",solver="sgd", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression15=MLPRegressor(activation="identity",solver="sgd", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression16=MLPRegressor(activation="logistic",solver="sgd", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression17=MLPRegressor(activation="logistic",solver="sgd", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression18=MLPRegressor(activation="logistic",solver="sgd", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression19=MLPRegressor(activation="tanh",solver="sgd", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression20=MLPRegressor(activation="tanh",solver="sgd", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression21=MLPRegressor(activation="relu",solver="sgd", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression22=MLPRegressor(activation="relu",solver="sgd", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression23=MLPRegressor(activation="relu",solver="sgd", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression24=MLPRegressor(activation="relu",solver="sgd", alpha=1e-7,hidden_layer_sizes=(20,1))     
        reqMLPregression25=MLPRegressor(activation="identity",solver="adam", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression26=MLPRegressor(activation="identity",solver="adam", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression27=MLPRegressor(activation="identity",solver="adam", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression28=MLPRegressor(activation="logistic",solver="adam", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression29=MLPRegressor(activation="logistic",solver="adam", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression30=MLPRegressor(activation="logistic",solver="adam", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression31=MLPRegressor(activation="tanh",solver="adam", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression32=MLPRegressor(activation="tanh",solver="adam", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression33=MLPRegressor(activation="tanh",solver="adam", alpha=1e-7,hidden_layer_sizes=(20,1))
        reqMLPregression34=MLPRegressor(activation="relu",solver="adam", alpha=1e-7,hidden_layer_sizes=(100,10))
        reqMLPregression35=MLPRegressor(activation="relu",solver="adam", alpha=1e-7,hidden_layer_sizes=(70,30))
        reqMLPregression36=MLPRegressor(activation="relu",solver="adam", alpha=1e-7,hidden_layer_sizes=(20,1))        

    if 'GaussianProcessRegressor_m' in model_name:
        kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                       1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
                       1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                            length_scale_bounds=(0.1, 10.0),
                                            periodicity_bounds=(1.0, 10.0)),
                       ConstantKernel(0.1, (0.01, 10.0))
                           * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
                       1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                    nu=1.5),DotProduct() + WhiteKernel()]
        gpr1 = GaussianProcessRegressor(kernel=kernels[0],random_state=0)
        gpr2 = GaussianProcessRegressor(kernel=kernels[1],random_state=0)
        gpr3 = GaussianProcessRegressor(kernel=kernels[2],random_state=0)
        gpr4 = GaussianProcessRegressor(kernel=kernels[3],random_state=0)
        gpr5 = GaussianProcessRegressor(kernel=kernels[4],random_state=0)
        gpr6 = GaussianProcessRegressor(kernel=kernels[5],random_state=0)

    if 'SGDRegressor_m' in model_name:
        regSGDRegressor_m1=linear_model.SGDRegressor(loss="squared_loss",max_iter=1000, tol=0.001)  
        regSGDRegressor_m2=linear_model.SGDRegressor(loss="huber",max_iter=1000, tol=0.001)       
        regSGDRegressor_m3=linear_model.SGDRegressor(loss="epsilon_insensitive",max_iter=1000, tol=0.001)       
        regSGDRegressor_m4=linear_model.SGDRegressor(loss="squared_epsilon_insensitive",max_iter=1000, tol=0.001)               
       

    X=X[list(list_variable)]    
    for k in range(1,nbre_fois):
        train_index=[]
        test_index=sample(list(X.index),perc)
        for k in list(X.index):
            if k not in test_index:
                train_index.append(k)
        X_test=X.iloc[test_index] # Chargement du X test
        y_test=Y.iloc[test_index] # Chargement du y test
    
        X_train=X.iloc[train_index] # Chargement du X train
        y_train=Y.iloc[train_index] # Chargement du y train    
        if 'RandomForestRegressor_m' in model_name:
            # arbre de decision
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m1)
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m2)
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m3)
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m4)
            RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m5)

        if 'treeDecision_regresion_m' in model_name:
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,regtreeDecision_regresion_m1)
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,regtreeDecision_regresion_m2)
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,regtreeDecision_regresion_m3)
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,regtreeDecision_regresion_m4)
            treeDecision_regresion_m(X_train, y_train,X_test, y_test,regtreeDecision_regresion_m5)
        if 'MLPregression' in model_name:
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression1)
            except:
                print("Toto")
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression2)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression3)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression4)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression5)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression6)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression7)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression8)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression9)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression10)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression11)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression12)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression13)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression14)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression15)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression16)
            except:
                pass 
            try:
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression17)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression18)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression19)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression20)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression21)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression22)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression23)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression24)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression25)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression26)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression27)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression28)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression29)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression30)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression31)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression32)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression33)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression34)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression35)
                MLPregression(X_train, y_train,X_test, y_test,reqMLPregression36)
            except:
                pass             
        if 'AdaBoostRegressor_m' in model_name:
                try:
                    AdaBoostRegressor_m(X_train, y_train,X_test, y_test,regAdaBoostRegressor_m)
                except:
                    pass
        if 'KNeighborsRegressor_m' in model_name:
            try:
                KNeighborsRegressor_m(X_train, y_train,X_test, y_test,regKNeighborsRegressor)
            except:
                pass
            
    
        "model lineaire"
        if 'linear_model_OLS' in model_name:
            try:
                "Ordinary Least Squares"
                linear_model_OLS(X_train, y_train,X_test, y_test,reglinearie)
            except:
                pass
        if 'linear_model_Ridge' in model_name:
            #for alph in [0.1,0.4,0.5,0.6,0.7,0.8]:
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge1)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge2)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge3)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge4)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge5)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge6)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge7)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge8)
            except:
                pass
            try:
                "Ordinary Least Rigde"
                linear_model_Ridge(X_train, y_train,X_test, y_test,reglinear_model_Ridge9)
            except:
                pass
        if 'BayesianRidge_m' in model_name: 
            try:
                BayesianRidge_m(X_train, y_train,X_test, y_test,regBayesianRidge_m_T)
            except:
                pass
            try:
                BayesianRidge_m(X_train, y_train,X_test, y_test,regBayesianRidge_m_F)
            except:
                pass
        if 'SGDRegressor_m' in model_name:        
            #for statut in ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"]:
            try:
                SGDRegressor_m(X_train, y_train,X_test, y_test,regSGDRegressor_m1)
            except:
                pass
            try:
                SGDRegressor_m(X_train, y_train,X_test, y_test,regSGDRegressor_m2)
            except:
                pass
            try:
                SGDRegressor_m(X_train, y_train,X_test, y_test,regSGDRegressor_m3)
            except:
                pass
            try:
                SGDRegressor_m(X_train, y_train,X_test, y_test,regSGDRegressor_m4)
            except:
                pass
            print("tot")
        
        if 'svm_m' in model_name:
            #"discrimation"
            try:
                svm_m(X_train, y_train,X_test, y_test,regvm_m1)
            except:
                pass
            try:
                svm_m(X_train, y_train,X_test, y_test,regvm_m2)
            except:
                pass

        if 'gaussian_process_m' in model_name:                
            #gaussian_process_m
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr1)
            except:
                pass    
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr2)
            except:
                pass       
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr3)
            except:
                pass  
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr4)
            except:
                pass    
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr5)
            except:
                pass       
            try:
                gaussian_process_m(X_train, y_train,X_test, y_test,gpr6)
            except:
                pass  
            #    try:
            #        exec("RandomForestRegressor_m(X_train, y_train,X_test, y_test,regRandomForestRegressor_m"+str(maxdep)+")")
            #    except:
            #        pass
#            #print(regRandomForestRegressor_m9)

#        print("toto")
#    

    

   
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