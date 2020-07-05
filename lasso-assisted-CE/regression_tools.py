# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:09:01 2019

@author: yifan
"""

'''
regression plot tools
'''
import os
import numpy as np
import json
from sklearn.metrics import mean_squared_error
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt 
#plt.switch_backend('agg')
font = {'family' : 'normal', 'size'   : 15}
matplotlib.rc('font', **font)


#%% User define functions 
from set_config_constants import Ec as Ec_init
from set_config_constants import config as config_init

def import_Xy(selected_batches = [0,1,2,3], fit_int_flag = False, iso_flag = True):
    '''
    %% Import data and get X and y
    '''
    
    
    n_all_batch = len(config_init)
    
    # read and save configuration and x into lists
    # include all isomorphs 
    x_batch = []
    config_batch = []
    Ec_batch = []
    
    if iso_flag:
        
        for batch_i in range(n_all_batch):
            
            # name of the json file
            json_name = 'ES_iso_' + str(batch_i) + '.json'
            # name of the pi file
            pi_name = 'pi_iso_' + str(batch_i) + '.npy'
            with open(json_name) as f:
                ES_data = json.load(f)
                
            Ec_batch_i = ES_data['E_iso']
            config_batch_i = ES_data['config_iso']
            
            config_batch.append(config_batch_i)
            Ec_batch.append(Ec_batch_i)
            x_batch.append(np.load(pi_name))
        
        '''
        select the batches for regression
        '''    
        
        x = x_batch[selected_batches[0]]
        config  = config_batch[selected_batches[0]]
        Ec = Ec_batch[selected_batches[0]]
        
        if len(selected_batches) > 1:
            
            for batch_i in selected_batches[1:]:
                x = np.concatenate((x, x_batch[batch_i]), axis = 0) # Concatenate the matrix row wise
                config =  config + config_batch[batch_i]
                Ec = Ec + Ec_batch[batch_i]
    
    else: 
        
        for batch_i in range(n_all_batch):
            config  = config + config_init[batch_i]
            Ec = Ec + Ec_init[batch_i]    
    
    # Prepare for the pi matrix - X
    X = x
    #
    # Add the first column of ones to pi matrix to fit intercept manually 
    if not fit_int_flag:
        X = np.ones((x.shape[0], x.shape[1]+1)) #the first column of pi matrix is set a 1, to be the intercept
        X[:,1:] = x     
    
    # Prepare for the true output values - y
    y = np.array(Ec)
    
    return X, y, config


'''
Plot the regression results
'''
      
def predict_y(x, intercept, J_nonzero):
    
    # x is the column in pi matrix or the pi matrix 
    y = np.dot(x, J_nonzero) + intercept
    # the results should be the same as y = lasso_cv.predict(X)
    return y


def cal_path(alphas, model, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag):
    
    '''
    Calculate both RMSE and number of coefficients path for plotting purpose
    '''
    
    RMSE_path = []
    coef_path = []
    
    for j in range(len(X_cv_train)):
        
        test_scores = np.zeros(len(alphas))
        coefs_i = np.zeros(len(alphas))
        
        print('{} % done'.format(100*(j+1)/len(X_cv_train)))
        
        for i, ai in enumerate(alphas):
            
            estimator = model(alpha = ai,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state = 0)
            estimator.fit(X_cv_train[j], y_cv_train[j])
            # Access the errors, error per cluster
            test_scores[i] = np.sqrt(mean_squared_error(y_cv_test[j], estimator.predict(X_cv_test[j]))) #RMSE
            coefs_i[i] = len(np.nonzero(estimator.coef_)[0])
        
        RMSE_path.append(test_scores)
        coef_path.append(coefs_i)
    
    RMSE_path = np.transpose(np.array(RMSE_path))
    coef_path = np.transpose(np.array(coef_path))

    
    return RMSE_path, coef_path



def plot_coef_path(alpha, alphas, coef_path, model_name, output_dir = os.getcwd()):
    '''
    #plot alphas vs the number of nonzero coefficents along the path
    '''


    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), coef_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(coef_path, axis = 1), 
             label='Average across the folds', linewidth=2)     
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    plt.legend(frameon=False, loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("# Nonzero Coefficients ")    
    plt.tight_layout()
    

    fig.savefig(os.path.join(output_dir, model_name + '_a_vs_n.png'))
    #plt.show() 



def plot_RMSE_path(alpha, alphas, RMSE_path, model_name, output_dir = os.getcwd()):
        
    '''
    #plot alphas vs RMSE along the path
    '''

    fig = plt.figure(figsize=(6, 4))
    
    plt.plot(-np.log10(alphas), RMSE_path, ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(RMSE_path, axis = 1), 
             label='Average across the folds', linewidth=2)  
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    
    plt.legend(frameon=False,loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("RMSE/cluster(ev)")    
    plt.tight_layout()
   
    fig.savefig(os.path.join(output_dir, model_name  + '_a_vs_cv.png'))
    #plt.show()   
       
def plot_path(X, y, n_sites, NPd_list, alpha, alphas, RMSE_path, coef_path, model, model_name, output_dir = os.getcwd()):
    
    '''
    Overall plot function for lasso/elastic net
    '''
    
    plot_coef_path(alpha, alphas, coef_path, model_name, output_dir)
    plot_RMSE_path(alpha, alphas, RMSE_path, model_name, output_dir)
    
    '''
    #make performance plot
    '''
    plot_performance(X, y, n_sites, NPd_list, model, model_name, output_dir)
    


def plot_ridge_path(alpha, alphas, RMSE_path, model_name, output_dir = os.getcwd()):
    
    fig = plt.figure(figsize=(6, 4))
    
    #plt.plot(-np.log10(alphas), np.log10(RMSE_path), ':', linewidth= 0.8)
    plt.plot(-np.log10(alphas), np.mean(RMSE_path, axis = 1), 
             label='Average across the folds', linewidth=2)  
    plt.axvline(-np.log10(alpha), linestyle='--' , color='r', linewidth=3,
                label='Optimal alpha') 
    
    plt.legend(frameon=False,loc='best')
    plt.xlabel(r'$-log10(\lambda)$')
    plt.ylabel("RMSE/cluster(ev)")    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name +'_a_vs_cv.png'))
    #plt.show()   

    
    
def plot_performance(X, y, n_sites, NPd_list, model, model_name, output_dir = os.getcwd()): 
    
    '''
    #plot parity plot
    '''
    y_predict_all = model.predict(X)
    #y_predict_all = predict_y(pi_nonzero, intercept, J_nonzero)
    
    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, y_predict_all, s=60, facecolors='none', edgecolors='r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Predicted Cluster Energy (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name + '_parity.png'))
    #plt.show()
    
    '''
    #plot error plot
    '''

    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, (y_predict_all - y)/n_sites, s = 20, color ='r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Error per lattice site (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, np.zeros(len(lims)), 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    #ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name +'_error_site.png'))
    #plt.show()
    
    '''
    #plot error plot per atom
    '''

    plt.figure(figsize=(6,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, (y_predict_all - y)/NPd_list, s=20, color = 'r')
    
    plt.xlabel("DFT Cluster Energy (eV)")
    plt.ylabel("Error per atom (eV)")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, np.zeros(len(lims)), 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    #ax.set_ylim(lims)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, model_name + '_error_atom.png'))
    #plt.show()

