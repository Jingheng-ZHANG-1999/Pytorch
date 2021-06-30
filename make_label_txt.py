# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:38:27 2020

@author: Zhang Jingheng
"""

with open("C:\\Users\\Zhang Jingheng\\Desktop\\file.txt","w") as f1:
    for i in range(0,100):
        i=str(i)
        f1.writelines("./isic/ISIC_00000"+i+".jpeg")
        f1.writelines("\n")
    for i in range(501,501):
        i=str(i)
        f1.writelines("./isic/ISIC_0000"+i+".jpeg")
        f1.writelines("\n")

