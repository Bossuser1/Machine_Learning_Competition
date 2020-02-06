#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:48:33 2020

@author: traore
"""
import os
from sklearn.externals import joblib




# Load the pickle file
clf_load = joblib.load('saved_model.pkl') 