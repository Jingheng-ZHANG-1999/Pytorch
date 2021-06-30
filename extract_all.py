# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:54:39 2020

@author: Zhang Jingheng
"""

import os
diagnosis = []
benign=r'"benign_malignant": "benign",'
malignant=r'"benign_malignant": "malignant",'
path1= "F:/Data/Descriptions/"    
                              
for i in range(11403,21800):    
    path2= str("ISIC_00"+str(i))
    filename= os.path.join(path1+path2)
    try:
      with open(filename,"r")as f:
        for line in f.readlines():
            line=line.strip()
            label0 = str(str(i)+"-0")
            label1 = str(str(i)+"-1")
            if line == benign:           
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\train.txt","a")as f1:
                  f1.writelines("F:/isic/train/ISIC_00"+str(i)+".jpeg "+"0")
                  f1.writelines("\n")                            
            elif line == malignant:
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\train.txt","a")as f1:
                  f1.writelines("F:/isic/train/ISIC_00"+str(i)+".jpeg "+"1")
                  f1.writelines("\n")  
    except: pass
        
for i in range(21800,24201):    
    path2= str("ISIC_00"+str(i))
    filename= os.path.join(path1+path2)
    try:
      with open(filename,"r")as f:
        for line in f.readlines():
            line=line.strip()
            label0 = str(str(i)+"-0")
            label1 = str(str(i)+"-1")
            if line == benign:           
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\test.txt","a")as f1:
                  f1.writelines("F:/isic/test/ISIC_00"+str(i)+".jpeg "+"0")
                  f1.writelines("\n")                            
            elif line == malignant:
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\test.txt","a")as f1:
                  f1.writelines("F:/isic/test/ISIC_00"+str(i)+".jpeg "+"1")
                  f1.writelines("\n")  
    except:pass
              
for i in range(24201,32000):    
    path2= str("ISIC_00"+str(i))
    filename= os.path.join(path1+path2)
    try:
      with open(filename,"r")as f:
        for line in f.readlines():
            line=line.strip()
            label0 = str(str(i)+"-0")
            label1 = str(str(i)+"-1")
            if line == benign:           
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\train.txt","a")as f1:
                  f1.writelines("F:/isic/train/ISIC_00"+str(i)+".jpeg "+"0")
                  f1.writelines("\n")                            
            elif line == malignant:
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\train.txt","a")as f1:
                  f1.writelines("F:/isic/train/ISIC_00"+str(i)+".jpeg "+"1")
                  f1.writelines("\n")  

    except:pass
    
for i in range(32000,34321):    
    path2= str("ISIC_00"+str(i))
    filename= os.path.join(path1+path2)
    try:
      with open(filename,"r")as f:
        for line in f.readlines():
            line=line.strip()
            label0 = str(str(i)+"-0")
            label1 = str(str(i)+"-1")
            if line == benign:           
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\test.txt","a")as f1:
                  f1.writelines("F:/isic/test/ISIC_00"+str(i)+".jpeg "+"0")
                  f1.writelines("\n")                            
            elif line == malignant:
              with open("C:\\Users\\Zhang Jingheng\\Desktop\\test.txt","a")as f1:
                  f1.writelines("F:/isic/test/ISIC_00"+str(i)+".jpeg "+"1")
                  f1.writelines("\n")  
    except:pass