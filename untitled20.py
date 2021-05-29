#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:08:22 2021

@author: vamshibukya
"""
from sklearn.preprocessing import Normalizer
import pickle

face_embedded=pickle.load(open('face_embedded.pickle','rb'))

x=face_embedded[0]

transformer=Normalizer().fit(x)

transformer.transform()



