#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:44:29 2020

@author: wtl
"""
import glob
import os
import shutil

dato = 'Week15'
pathname = os.path.join('/Users/julielehn-schioler/Desktop/EEG-patienter/', dato)

# Simple function which goes through all files in a folder and sorts the
# based upon the patients ID

for file in glob.glob(os.path.join(pathname, '*')):
    
    # The ID is isolated from the pathname; simple for most files 
    # but with an extra step for teh rest

    patient = file[len(pathname)+30:-4]
    
    if not patient.isdigit():
        if patient[0] is ' ':
            patient = patient[1:4]
        else:    
            patient = patient[:3]
    
    #print('file:', file)
    #print('patient:', patient)
    
    folder = (os.path.join(pathname, 'p' + str(patient)))
    
    # if the respective patient folder already exists, the file is moved there
    # else the folder is created and the files moved
    
    if not os.path.exists(folder):
        os.makedirs(folder)
       
    shutil.move(file, folder)    