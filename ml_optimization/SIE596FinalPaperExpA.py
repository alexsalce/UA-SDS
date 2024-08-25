# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:17:19 2024

@author: Alex Salce
"""
# import SIE596FinalAlgs
from SIE596FinalAlgs import gdDataADAM, gdDataSARAH, objective_function
import numpy as np
import matplotlib.pyplot as plt
import math
import time



#   ______                                   __      __                               
#  /      \                                 |  \    |  \                              
# |  $$$$$$\ __    __  _______    _______  _| $$_    \$$  ______   _______    _______ 
# | $$_  \$$|  \  |  \|       \  /       \|   $$ \  |  \ /      \ |       \  /       \
# | $$ \    | $$  | $$| $$$$$$$\|  $$$$$$$ \$$$$$$  | $$|  $$$$$$\| $$$$$$$\|  $$$$$$$
# | $$$$    | $$  | $$| $$  | $$| $$        | $$ __ | $$| $$  | $$| $$  | $$ \$$    \ 
# | $$      | $$__/ $$| $$  | $$| $$_____   | $$|  \| $$| $$__/ $$| $$  | $$ _\$$$$$$\
# | $$       \$$    $$| $$  | $$ \$$     \   \$$  $$| $$ \$$    $$| $$  | $$|       $$
#  \$$        \$$$$$$  \$$   \$$  \$$$$$$$    \$$$$  \$$  \$$$$$$  \$$   \$$ \$$$$$$$ 
                                                                                    
                                                                                    
                                                                                    

def plotnorm(x,xprev,alpha):
    return np.linalg.norm(x-xprev)/alpha


#   __            __               
#  /\ \          /\ \__            
#  \_\ \     __  \ \ ,_\    __     
#  /'_` \  /'__`\ \ \ \/  /'__`\   
# /\ \L\ \/\ \L\.\_\ \ \_/\ \L\.\_ 
# \ \___,_\ \__/.\_\\ \__\ \__/.\_\
#  \/__,_ /\/__/\/_/ \/__/\/__/\/_/
                                 
                                 
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

from ucimlrepo import fetch_ucirepo 

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
A = wine_quality.data.features 
b = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

A = np.array(A)
# A = np.append(A,np.ones([A.shape[0],1]), axis=1)
b = np.array(b)
x = np.zeros((A.shape[1],1))
n,m = A.shape
xstar = np.load("xstar_winedata.npy")
xstar = np.expand_dims(xstar,axis=1)



                                 
#  _ | _ |_  _ |   _  _  _ _  _  _ 
# (_)|(_)|_)(_||  |_)(_|| (_||||_) 
# _/              |                

n,m = A.shape

#SETTINGS FOR GRAPHS A  
epochs = 10

# #SETTINGS FOR GRAPHS B 
# epochs=60



#  _  _    
# ( \/ )\|/
#  )  ( /|\
# (_/\_)   

# NOTE: x* is computed using scipy.optimize.lsq_linear| with tolerance set to 1e-10
# function in attached xstar.py file and imported to this environment


#                 __                  ___                         
#  __          __/\ \__  __          /\_ \    __                  
# /\_\    ___ /\_\ \ ,_\/\_\     __  \//\ \  /\_\  ____      __   
# \/\ \ /' _ `\/\ \ \ \/\/\ \  /'__`\  \ \ \ \/\ \/\_ ,`\  /'__`\ 
#  \ \ \/\ \/\ \ \ \ \ \_\ \ \/\ \L\.\_ \_\ \_\ \ \/_/  /_/\  __/ 
#   \ \_\ \_\ \_\ \_\ \__\\ \_\ \__/.\_\/\____\\ \_\/\____\ \____\
#    \/_/\/_/\/_/\/_/\/__/ \/_/\/__/\/_/\/____/ \/_/\/____/\/____/                                                            
#  ____    ______  ____    ______  __  __     
# /\  _`\ /\  _  \/\  _`\ /\  _  \/\ \/\ \    
# \ \,\L\_\ \ \L\ \ \ \L\ \ \ \L\ \ \ \_\ \   
#  \/_\__ \\ \  __ \ \ ,  /\ \  __ \ \  _  \  
#    /\ \L\ \ \ \/\ \ \ \\ \\ \ \/\ \ \ \ \ \ 
#    \ `\____\ \_\ \_\ \_\ \_\ \_\ \_\ \_\ \_\
#     \/_____/\/_/\/_/\/_/\/ /\/_/\/_/\/_/\/_/
                                            
#SETTINGS FOR GRAPHS A                                          
epoch_split = 0.5

# #SETTINGS FOR GRAPHS B                                    
# epoch_split = 30

outer_loop_iter = int(epochs/epoch_split)
inner_loop_iter = int(epoch_split*n) # Number of iterations

vec = []
batchsize_sarah = 1
epsilon_sarah = 1e-2


SARAHdata = gdDataSARAH(A, b, outer_loop_iter, inner_loop_iter, batchsize_sarah)

eta_sarah = 1/(2.1*SARAHdata.LipLS())

mu = SARAHdata.muLS()
sigmam = (1/(mu * eta_sarah* (inner_loop_iter+1))) + (eta_sarah * SARAHdata.LipLS() / (2-(eta_sarah*SARAHdata.LipLS())))  #SARAH sconvex


kappa = SARAHdata.LipLS()/mu

#OPTIONAL FOR STRONG CONVEX CASE EPSILON CONVERGENCE *usually impractical)
# outer_loop_iter = math.ceil(np.log(np.linalg.norm(SARAHdata.GradLS(x))**2/epsilon_sarah/np.log(9/7)))
# inner_loop_iter = math.ceil(4.5*kappa)



#  ____    ______  ____    ______  __  __     
# /\  _`\ /\  _  \/\  _`\ /\  _  \/\ \/\ \    
# \ \,\L\_\ \ \L\ \ \ \L\ \ \ \L\ \ \ \_\ \   
#  \/_\__ \\ \  __ \ \ ,  /\ \  __ \ \  _  \  
#    /\ \L\ \ \ \/\ \ \ \\ \\ \ \/\ \ \ \ \ \ 
#    \ `\____\ \_\ \_\ \_\ \_\ \_\ \_\ \_\ \_\
#     \/_____/\/_/\/_/\/_/\/ /\/_/\/_/\/_/\/_/
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

np.random.seed(123)
myData = SARAHdata
P_wstar = (1/myData.n)*myData.GradLS(xstar)
wold = np.copy(x)
w_list_sarah = []
w_list = []
sarah_obj_plot = []
sarah_steps_plot = []
sarah_gradnorm_plot = []

sarah_start = time.time()

for _ in range(outer_loop_iter):
    
    w = wold
    vold = (1/myData.n)*myData.GradLS(w)
    w = wold - eta_sarah * vold
    
    for i in range(inner_loop_iter):
        i = np.random.permutation(myData.n)[:myData.batch]
        vnew = myData.sGradLS(w, i) - myData.sGradLS(wold, i) + vold
        wold = np.copy(w)
        w_list_sarah.append(wold)   #THIS IS JUST FOR PLOT
        w_list.append(wold)
        w = w - eta_sarah * vnew
        vold = np.copy(vnew)
        sarah_obj_plot.append(myData.objective_function_ls(w))
        sarah_steps_plot.append(plotnorm(w, wold, eta_sarah))
        
    
    wold = w_list[ np.random.permutation(range(0,len(w_list)))[0] ]
    w_list = []

sarah_end = time.time()
sarah_time = sarah_end - sarah_start
print(sarah_time, " seconds to run SARAH")    

# for i in range(len(w_list_sarah)):
#     sarah_gradnorm_plot.append(np.linalg.norm((1/myData.n)*myData.GradLS(w_list_sarah[i])-P_wstar))

for i in range(len(w_list_sarah)):
    sarah_gradnorm_plot.append(np.linalg.norm(w_list_sarah[i]-xstar))


#         _       _          _       _           _        
#        / /\    /\ \    _ / /\     /\ \        /\ \      
#       / /  \   \ \ \  /_/ / /    /  \ \      /  \ \     
#      / / /\ \__ \ \ \ \___\/    / /\ \ \    / /\ \_\    
#     / / /\ \___\/ / /  \ \ \   / / /\ \_\  / / /\/_/    
#     \ \ \ \/___/\ \ \   \_\ \ / / /_/ / / / / / ______  
#      \ \ \       \ \ \  / / // / /__\/ / / / / /\_____\ 
#  _    \ \ \       \ \ \/ / // / /_____/ / / /  \/____ / 
# /_/\__/ / /        \ \ \/ // / /\ \ \  / / /_____/ / /  
# \ \/___/ /          \ \  // / /  \ \ \/ / /______\/ /   
#  \_____\/            \_\/ \/_/    \_\/\/___________/    
                                                        
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

np.random.seed(123)
myData = SARAHdata
P_wstar = (1/myData.n)*myData.GradLS(xstar)
eta_svrg = 1/(5*SARAHdata.LipLS())

alpham = (1/(mu * eta_svrg* (1-2*SARAHdata.LipLS())*inner_loop_iter)) + (2*eta_svrg * SARAHdata.LipLS() / (1-2*(eta_svrg*SARAHdata.LipLS())))

wold = np.copy(x)
w_list_svrg = []
w_list = []
svrg_obj_plot = []
svrg_steps_plot = []
svrg_gradnorm_plot = []

svrg_start = time.time()

for _ in range(outer_loop_iter):
    
    w = wold
    vold = (1/myData.n)*myData.GradLS(w)
    # w = wold - eta_svrg * vold
    w0 = np.copy(w)
    
    for i in range(inner_loop_iter):
        i = np.random.permutation(myData.n)[:myData.batch]
        vnew = myData.sGradLS(w, i) - myData.sGradLS(w0, i) + vold
        wold = np.copy(w)
        w_list_svrg.append(wold)   #THIS IS JUST FOR PLOT
        w_list.append(wold)
        w = w - eta_svrg * vnew
        svrg_obj_plot.append(myData.objective_function_ls(w))
        svrg_steps_plot.append(plotnorm(w, wold, eta_svrg))
    
    wold = w_list[ np.random.permutation(range(0,len(w_list)))[0] ]
    w_list = []

svrg_end = time.time()
svrg_time = svrg_end - svrg_start
print(svrg_time, " seconds to run SVRG")          
        
# for i in range(len(w_list_svrg)):
#     svrg_gradnorm_plot.append(np.linalg.norm((1/myData.n)*myData.GradLS(w_list_svrg[i])-P_wstar))

for i in range(len(w_list_svrg)):
    svrg_gradnorm_plot.append(np.linalg.norm(w_list_svrg[i]-xstar))

#                 __                  ___                         
#  __          __/\ \__  __          /\_ \    __                  
# /\_\    ___ /\_\ \ ,_\/\_\     __  \//\ \  /\_\  ____      __   
# \/\ \ /' _ `\/\ \ \ \/\/\ \  /'__`\  \ \ \ \/\ \/\_ ,`\  /'__`\ 
#  \ \ \/\ \/\ \ \ \ \ \_\ \ \/\ \L\.\_ \_\ \_\ \ \/_/  /_/\  __/ 
#   \ \_\ \_\ \_\ \_\ \__\\ \_\ \__/.\_\/\____\\ \_\/\____\ \____\
#    \/_/\/_/\/_/\/_/\/__/ \/_/\/__/\/_/\/____/ \/_/\/____/\/____/
#  ______  ____    ______              
# /\  _  \/\  _`\ /\  _  \  /'\_/`\    
# \ \ \L\ \ \ \/\ \ \ \L\ \/\      \   
#  \ \  __ \ \ \ \ \ \  __ \ \ \__\ \  
#   \ \ \/\ \ \ \_\ \ \ \/\ \ \ \_/\ \ 
#    \ \_\ \_\ \____/\ \_\ \_\ \_\\ \_\
#     \/_/\/_/\/___/  \/_/\/_/\/_/ \/_/


alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon_adam = 1e-8
batchsize_adam = 10
max_iter_adam = epochs*n


ADAMdata = gdDataADAM(A, b,max_iter_adam,alpha,beta1,beta2,batchsize_adam,epsilon_adam)


#  ______  ____    ______              
# /\  _  \/\  _`\ /\  _  \  /'\_/`\    
# \ \ \L\ \ \ \/\ \ \ \L\ \/\      \   
#  \ \  __ \ \ \ \ \ \  __ \ \ \__\ \  
#   \ \ \/\ \ \ \_\ \ \ \/\ \ \ \_/\ \ 
#    \ \_\ \_\ \____/\ \_\ \_\ \_\\ \_\
#     \/_/\/_/\/___/  \/_/\/_/\/_/ \/_/
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

adam_start = time.time()

np.random.seed(123)
myData = ADAMdata
P_wstar = (1/myData.n)*myData.GradLS(xstar)
w = np.copy(x)
m = np.copy(x)
v = np.copy(x)
t = 0
w_list_adam = []
adam_obj_plot = []
adam_steps_plot = []
adam_gradnorm_plot = []


for _ in range(myData.maxit):
    t += 1
    i = np.random.permutation(myData.n)[:myData.batch]
    g = myData.sGradLS(w,i)   #update batch gradient
    m = myData.beta1 * m + (1 - myData.beta1) * g    #update biased first moment est
    v = myData.beta2 * v + (1 - myData.beta2) * g**2    #update biased second raw moment est
    mhat = m/(1 - myData.beta1**t)  #bias correction first moment est
    vhat = v/(1 - myData.beta2**t)  #bias correction second moment est
    wold = np.copy(w)
    w = w - ( alpha * mhat / (np.sqrt(vhat) + epsilon_adam) )
    
    #stash for graphs
    w_list_adam.append(w)
    adam_steps_plot.append(plotnorm(w, wold, alpha))
    adam_obj_plot.append(objective_function(w,A,b))
    

adam_end = time.time()
adam_time = adam_end - adam_start
print(adam_time, " seconds to run ADAM")          
    
# for i in range(len(w_list_adam)):
#     adam_gradnorm_plot.append(np.linalg.norm((1/myData.n)*myData.GradLS(w_list_adam[i])-P_wstar))

for i in range(len(w_list_adam)):
    adam_gradnorm_plot.append(np.linalg.norm(w_list_adam[i]-xstar))

#  ____    __       _____   ______  ____       
# /\  _`\ /\ \     /\  __`\/\__  _\/\  _`\     
# \ \ \L\ \ \ \    \ \ \/\ \/_/\ \/\ \,\L\_\   
#  \ \ ,__/\ \ \  __\ \ \ \ \ \ \ \ \/_\__ \   
#   \ \ \/  \ \ \L\ \\ \ \_\ \ \ \ \  /\ \L\ \ 
#    \ \_\   \ \____/ \ \_____\ \ \_\ \ `\____\
#     \/_/    \/___/   \/_____/  \/_/  \/_____/
                              
final = [w_list_sarah[-1],w_list_svrg[-1],w_list_adam[-1],xstar]
opt_obj_plot = objective_function(xstar,A,b)*np.ones(len(sarah_obj_plot))
opt_obj_plot = opt_obj_plot.tolist()

#GRAPHS A
plotint = math.ceil(n/10)

#GRAPHS B
# plotint = math.ceil(n)

epoch_indices = []
for i in range(1,epochs):
    epoch_indices.append(i*n)


ymax = 1000

plt.plot(sarah_steps_plot[::plotint],'-b',label='SARAH')
plt.plot(svrg_steps_plot[::plotint], '-r',label='SVRG')
plt.plot(adam_steps_plot[::plotint],'-g',label='ADAM')
ax = plt.gca()
ax.set_ylim([0, ymax])
plt.legend()
plt.title(label='||x_t-x_t-1|| vs epochs')
#GRAPHS A
plt.xlabel("epoch*10")
#GRAPHS B
# plt.xlabel("epoch")
plt.show()

ymin = 0.95*np.min([np.min(adam_obj_plot),np.min(sarah_obj_plot),np.min(svrg_obj_plot)])
ymax = 0.3*np.min([np.max(adam_obj_plot),np.max(sarah_obj_plot),np.max(svrg_obj_plot)])

plt.plot(sarah_obj_plot[::plotint],'-b',label='SARAH')
plt.plot(svrg_obj_plot[::plotint], '-r',label='SVRG')
plt.plot(adam_obj_plot[::plotint],'-g',label='ADAM')
# plt.plot(opt_obj_plot[::plotint],'-y',label='x*')
ax = plt.gca()
ax.set_ylim([ymin, ymax])
plt.legend()
plt.title(label='||Ax_t-b||^2 vs epochs')
#GRAPHS A
plt.xlabel("epoch*10")
#GRAPHS B
# plt.xlabel("epoch")
plt.show()

plt.plot(sarah_gradnorm_plot[::plotint],'-b',label='SARAH')
plt.plot(svrg_gradnorm_plot[::plotint], '-r',label='SVRG')
plt.plot(adam_gradnorm_plot[::plotint],'-g',label='ADAM')
plt.legend()
plt.title(label='||x_t-x*|| vs epochs')
#GRAPHS A
plt.xlabel("epoch*10")
#GRAPHS B
# plt.xlabel("epoch")
plt.show()
