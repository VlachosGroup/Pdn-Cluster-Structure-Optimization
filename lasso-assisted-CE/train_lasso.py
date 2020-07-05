# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:37:24 2018

@author: wangyf
"""


import lattice_functions as lf
import regression_tools as rtools
from set_ce_lattice import mother

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV, lasso_path, Lasso
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

import numpy as np
import pickle 
import json
import os

model_name = 'lasso'
base_dir = os.getcwd()

#%% Import data
'''
fit_int_flag, explained: 
    
True - fit intercept automatically from LASSO regression
     - the intercept is obtained from LASSO.intercept_
False- fit intercept by adding additional first column of pi matrix 
     - the intercept is obtained from the first cluster interaction
     - the signficant cluster interactions need to be adjusted based on index
'''

fit_int_flag = False
iso_flag = True

selected_batches = [0,1,2, 3]

X_init, y, config = rtools.import_Xy(selected_batches  , fit_int_flag, iso_flag)
# the number of Pd atoms in each structure
NPd_list = lf.get_NPd_list(config)
n_sites = len(mother)

# Load clusters
with open('clusters.json') as f:
    Gcv = json.load(f)['Gcv']  


# Modify the model name with selected dataset
model_name = model_name + '_' + ''.join(str(i) for i in selected_batches) 
output_dir = os.path.join(base_dir, model_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)    

#%% The date set is needed to be standardized for lasso, except for the intercept
X = X_init.copy()
if not fit_int_flag:
    scaler = StandardScaler().fit(X[:,1:])
    X[:,1:] = scaler.transform(X[:,1:])
else: 
    scaler = StandardScaler().fit(X)
    X= scaler.transform(X)
    
    
sv = scaler.scale_ # standard deviation for each x variable
mv = scaler.mean_ # mean for each x variable

n_unfilled_clusters = len(np.nonzero(mv)[0]) # the number of clusters that have not been filled

#%% Preparation before regression
# Train test split, save 10% of data point to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

NPd_test = []
NPd_train = []
for i in y_test: NPd_test.append(NPd_list[np.where(y==i)[0][0]])
for i in y_train: NPd_train.append(NPd_list[np.where(y==i)[0][0]])
                               
NPd_test = np.array(NPd_test)
NPd_train = np.array(NPd_train)                              

# The alpha grid used for plotting path
alphas_grid = np.logspace(0, -3, 20)

# Cross-validation scheme                                  
rkf = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 0)

# Explicitly take out the train/test set
X_cv_train, y_cv_train, X_cv_test, y_cv_test = [],[],[],[]
for train_index, test_index in rkf.split(X_train):
    X_cv_train.append(X_train[train_index])
    y_cv_train.append(y_train[train_index])
    X_cv_test.append(X_train[test_index])
    y_cv_test.append(y_train[test_index])

#%% LASSO regression
'''   
# LassoCV to obtain the best alpha, the proper training of Lasso
'''
lasso_cv  = LassoCV(cv = rkf,  max_iter = 1e7, tol = 0.001, fit_intercept=fit_int_flag, random_state=0)
lasso_cv.fit(X_train, y_train)

# the optimal alpha from lassocv
lasso_alpha = lasso_cv.alpha_
# Coefficients for each term
lasso_coefs = lasso_cv.coef_
# The original intercepts 
lasso_intercept = lasso_cv.intercept_

# Access the errors 
y_predict_test = lasso_cv.predict(X_test)
y_predict_train = lasso_cv.predict(X_train)

# error per cluster per site
lasso_RMSE_test_site = np.sqrt(mean_squared_error(y_test, y_predict_test))/n_sites
lasso_RMSE_train_site = np.sqrt(mean_squared_error(y_train, y_predict_train))/n_sites

# error per atom in the cluster
lasso_RMSE_test_atom = np.sqrt(mean_squared_error(y_test/NPd_test, y_predict_test/NPd_test))
lasso_RMSE_train_atom = np.sqrt(mean_squared_error(y_train/NPd_train, y_predict_train/NPd_train))



##Use alpha grid prepare for lassopath
lasso_RMSE_path, lasso_coef_path = rtools.cal_path(alphas_grid, Lasso, X_cv_train, y_cv_train, X_cv_test, y_cv_test, fit_int_flag)
##lasso_path to get alphas and coef_path, somehow individual CV does not work
#lasso_alphas, lasso_coef_path, _ = lasso_path(X_train, y_train, alphas = alphas_grid, fit_intercept=fit_int_flag)
rtools.plot_path(X, y, n_sites, NPd_list, lasso_alpha, alphas_grid, lasso_RMSE_path, lasso_coef_path, lasso_cv, model_name, output_dir)

'''
lasso coefficients needed to be tranformed back to the regular form
'''
lasso_coefs_regular = np.zeros(len(lasso_coefs))
lasso_coefs_regular[1:] = lasso_coefs[1:]/sv
lasso_coefs_regular[0] = lasso_coefs[0] - np.sum(mv/sv*lasso_coefs[1:])

#%% Select the significant cluster interactions 
'''
LASSO Post Processing
'''

# Set the tolerance for signficant interactions 
Tol = 1e-5
# The indices for non-zero coefficients/significant cluster interactions 
J_index = np.where(abs(lasso_coefs_regular)>Tol)[0]
# The number of non-zero coefficients/significant cluster interactions  
n_nonzero = len(J_index)
# The values of non-zero coefficients/significant cluster interactions  
J_nonzero = lasso_coefs_regular[J_index] 
pi_nonzero = X[:, J_index]

# Pick the significant clusters
Gcv_nonzero = []

# Adjust for the manual intercept fitting
if not fit_int_flag:
    
    intercept = J_nonzero[0]
    n_nonzero = n_nonzero - 1 
    J_nonzero = J_nonzero[1:]
    pi_nonzero = pi_nonzero[:,1:]

    
    for i in J_index[1:]:
        # take out the first one and adjust the indices by -1 
        Gcv_nonzero.append(Gcv[i-1]) 
else:       
    for i in J_index:
        Gcv_nonzero.append(Gcv[i]) 

'''
Save Gcv_nonzero and J_nonzero to pickle for further use
''' 
pickle.dump([Gcv_nonzero, J_nonzero, intercept, lasso_RMSE_test_atom, lasso_RMSE_test_site], 
            open(os.path.join(output_dir, model_name + '.p'),'wb'))
 			
			
#%% Energy test for lasso
from ase.io import read, write
def save_POV(atoms, index, output_dir):

    pov_args = {
    	'transparent': True, #Makes background transparent. I don't think I've had luck with this option though
        'canvas_width': 1800., #Size of the width. Height will automatically be calculated. This value greatly impacts POV-Ray processing times
        'display': False, #Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '0x, 0y,0z', #Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        'celllinewidth': 0.02, #Thickness of cell lines
        'show_unit_cell': 0 #Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        #You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        #e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
        }

    #Write to POV-Ray file
    filename = 'Pd'+'-' + str(index) + '.POV'
    if not os.path.exists(output_dir): os.makedirs(output_dir)    
    write(os.path.join(output_dir, filename), atoms, **pov_args)
    
    
       
#%% Seaborn lasso join plot
import matplotlib.pyplot as plt 
import seaborn as sns;   

y_predict_all = lasso_cv.predict(X)


lims = [
    np.min([y.min(), y_predict_all.min()]),  # min of both axes
    np.max([y.max(), y_predict_all.max()]),  # max of both axes
]
#    
fig = plt.figure(figsize=(10, 6))
sns.set(rc={'figure.figsize':(10,6)})
sns.set(font_scale=1.4)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})    
g = sns.JointGrid(y, y_predict_all)
g = g.set_axis_labels("DFT Cluster Energy (eV)", "Predicted Cluster Energy (eV)")

g = g.plot_joint(plt.scatter, color="b", edgecolors = 'navy', s=60, alpha = 0.1) #, facecolors='white', edgecolors = 'b'
g.ax_joint.plot(lims, lims, '--k')

_ = g.ax_marg_x.scatter( y, (y_predict_all - y)/n_sites, s = 30, color ='b', edgecolors = 'navy', alpha = 0.1 ) #facecolors='white', edgecolors = 'b',
_ = g.ax_marg_y.hist(y_predict_all, color="b", alpha = 0.8, orientation="horizontal", bins=np.linspace(lims[0], lims[1], 12))
g.ax_marg_x.plot(lims, np.zeros(len(lims)), 'k--')
g.ax_marg_x.set_ylim([-0.02,0.02])

g.ax_marg_x.set_ylabel("Error/site (eV)")
g.ax_marg_y.set_xlabel("Frequency")
    
plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
plt.setp(g.ax_marg_y.get_xticklabels(), visible=True)    
plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=True)
plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=True)
plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=True)
plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=True)
plt.show()
#fig.savefig(os.path.join(output_dir, model_name + '_combo.png'))