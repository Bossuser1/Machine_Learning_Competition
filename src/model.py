#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:22:56 2020

@author: traore
"""

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
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

#Supervised learning


def linear_model_OLS(X_train, y_train,X_test, y_test):
    "Ordinary Least Squares"
    try:
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        print("Linear model Ordinary Least Squares")
        yTest_predicted =reg.predict(X_test)
        print(reg.score(X_test, y_test))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
    except:
        pass


def linear_model_Ridge(X_train, y_train,X_test, y_test,alph):
    "Ridge"
    try:
        reg = linear_model.Ridge(alpha=alph)
        reg.fit(X_train, y_train)
        print("Linear model Ridge"+str(alph))
        yTest_predicted =reg.predict(X_test)
        print(reg.score(X_test, y_test))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
    except:
        pass

def BayesianRidge_m(X_train, y_train,X_test, y_test,statut):
    "Ridge"
    try:
        reg=linear_model.BayesianRidge(compute_score=statut)
        reg.fit(X_train, y_train)
        print("BayesianRidge_m "+str(statut))
        yTest_predicted =reg.predict(X_test)
        print(reg.score(X_test, y_test))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
    except:
        pass

def SGDRegressor_m(X_train, y_train,X_test, y_test,statut):
    "SGDRegressor"
    try:
        reg=linear_model.SGDRegressor(loss=statut,max_iter=1000, tol=0.001)
        reg.fit(X_train, y_train)
        print("SGDRegressor_m "+str(statut))
        yTest_predicted =reg.predict(X_test)
        print(reg.score(X_test, y_test))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
    except:
        pass


def svm_m(X_train, y_train,X_test, y_test,statut,ty):
        "svm"
        try:
            if ty=="svm":
                clf_SVR = svm.SVR()
            elif ty=="LinearSVR":
                clf_SVR = svm.LinearSVR()
            else:
                pass
            clf_SVR.fit(X_train, y_train)
            score_SVR = clf_SVR.score(X_test, y_test)
            print(" SVM "+statut+" "+ty)
            print(score_SVR)
            y_pred=clf_SVR.predict(X_test)
            print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
        except:
            pass     


def gaussian_process_m(X_train, y_train,X_test, y_test):
    "gaussian_process"
    kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
               1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
               1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                    length_scale_bounds=(0.1, 10.0),
                                    periodicity_bounds=(1.0, 10.0)),
               ConstantKernel(0.1, (0.01, 10.0))
                   * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
               1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                            nu=1.5),DotProduct() + WhiteKernel()]
    for k in range(len(kernels)):
        try:
            gpr = GaussianProcessRegressor(kernel=kernels[k],random_state=0).fit(X_train, y_train)
            print("gaussian_process "+str(k))
            score_gpr = gpr.score(X_test, y_test)
            print(score_gpr)
            y_pred=gpr.predict(X_test)
            print("RMSE:"+str(mean_squared_error(y_test, y_pred)))        
        except:
            pass        


#def naiveBaye_m(X_train, y_train,X_test, y_test):
#    "naaive bayes"

def treeDecision_regresion_m(X_train, y_train,X_test, y_test,maxt):
    #tree Descion
    try:
        clf_tree = tree.DecisionTreeRegressor(max_depth=maxt)
        clf_tree.fit(X_train, y_train)
        score_tree = clf_tree.score(X_test, y_test)
        print(" arbre de decision"+str(maxt))
        print(score_tree)
        y_pred=clf_tree.predict(X_test)
        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
    except:
        pass    
    
# randomForestRegression
        
def RandomForestRegressor_m(X_train, y_train,X_test, y_test,maxt):
    #tree Descion
    try:
        clf_tree = RandomForestRegressor(max_depth=maxt, random_state=0)
        clf_tree.fit(X_train, y_train)
        score_tree = clf_tree.score(X_test, y_test)
        print(" random forest "+str(maxt))
        print(score_tree)
        y_pred=clf_tree.predict(X_test)
        print("RMSE:"+str(mean_squared_error(y_test, y_pred)))
    except:
        pass 
    



def MLPregression(X_train, y_train,X_test, y_test,solver,size_hidden,activation1):
    try:
        # version classification
        if solver in ["lbfgs","sgd","adam"]:
            modelMLPR = MLPRegressor(activation=activation1,solver=solver, alpha=1e-7,hidden_layer_sizes=size_hidden,)
        else:
            solver='lbfgs'
            modelMLPR = MLPRegressor(activation=activation1,solver=solver, alpha=1e-7,hidden_layer_sizes=size_hidden)
        #to learn
        modelMLPR.fit(X_train, y_train)
        #to predict
        print("Neural network "+str(solver)+" "+str(activation1))
        yTest_predicted =modelMLPR.predict(X_test)
        print(modelMLPR.score(X_test, y_test))
        print("RMSE:"+str(mean_squared_error(y_test, yTest_predicted)))
    except:
        pass
