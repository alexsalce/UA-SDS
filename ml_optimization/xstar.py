# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 07:35:09 2024

@author: Alex Salce
"""

from ucimlrepo import fetch_ucirepo 
import numpy as np
from scipy.optimize import lsq_linear
#           _                                _ _ _               _       _        
# __      _(_)_ __   ___    __ _ _   _  __ _| (_| |_ _   _    __| | __ _| |_ __ _ 
# \ \ /\ / | | '_ \ / _ \  / _` | | | |/ _` | | | __| | | |  / _` |/ _` | __/ _` |
#  \ V  V /| | | | |  __/ | (_| | |_| | (_| | | | |_| |_| | | (_| | (_| | || (_| |
#   \_/\_/ |_|_| |_|\___|  \__, |\__,_|\__,_|_|_|\__|\__, |  \__,_|\__,_|\__\__,_|
#                             |_|                    |___/                        

###############################################################################
#CITATION

# Cortez,Paulo, Cerdeira,A., Almeida,F., Matos,T., and Reis,J.. (2009). 
# Wine Quality. UCI Machine Learning Repository. 
# https://doi.org/10.24432/C56S3T.
###############################################################################

# # fetch dataset 
# wine_quality = fetch_ucirepo(id=186) 
  
# # data (as pandas dataframes) 
# A = wine_quality.data.features 
# b = wine_quality.data.targets 
  
# # metadata 
# print(wine_quality.metadata) 
  
# # variable information 
# print(wine_quality.variables) 

# A = np.array(A)
# # A = np.append(A,np.ones([A.shape[0],1]), axis=1)
# b = np.array(b)
# b = np.squeeze(b)
# # x = np.zeros((A.shape[1],1))
# n,m = A.shape



#        _                                           
#  _    |_) _  __ _  _ __  _|    _ _|_ o     o _|_ \/
# _> |_||  (/_ | (_ (_)| |(_||_|(_  |_ | \_/ |  |_ / 
                                                   
#  _| _ _|_ _                                        
# (_|(_| |_(_|                                       

# ###############################################################################
# #CITATION
 
# # Hamidieh,Kam. (2018). Superconductivty Data. UCI Machine Learning Repository. 
# # https://doi.org/10.24432/C53P47.

# ###############################################################################

  
# fetch dataset 
superconductivty_data = fetch_ucirepo(id=464) 
  
# data (as pandas dataframes) 
A = superconductivty_data.data.features 
b = superconductivty_data.data.targets 
  
# metadata 
print(superconductivty_data.metadata) 
  
# variable information 
print(superconductivty_data.variables) 


A = np.array(A)
A = np.append(A,np.ones([A.shape[0],1]), axis=1)
b = np.array(b)
b = np.squeeze(b)
x = np.zeros((A.shape[1],1))


model = lsq_linear(A, b,tol=1e-10, lsq_solver='exact',method='bvls')
xstar = model.x


