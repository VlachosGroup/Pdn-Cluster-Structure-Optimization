# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:04:45 2018

@author: wangyf
"""

from ase.visualize import view
from ase.build import fcc111
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dpi = 300.

element = 'Pd'
size = (4,3,3)
atoms = fcc111(element,size)
i = int(len(atoms)/2)
#view(atoms)

                               
                               
dist = atoms.get_distances(i,range(0,len(atoms)))
dist_t = np.around(dist, decimals = 3)
dist_sort =  np.sort(dist_t)
dist_v = np.array(list(set(dist_sort)))
NN_dist = dist_v[1:]
NN_dist_scaled = NN_dist/NN_dist[0]


CI1 = 0.35
NN1 = (NN_dist[0] - CI1, NN_dist[0] + CI1)
CI2 = 0.25
NN2 = (NN_dist[1] - CI1, NN_dist[1] + CI1)

x = atoms.positions[:,0]
y = atoms.positions[:,1]
z = atoms.positions[:,2]
#%%
def plot_NN():
    cm = ['black', 'dimgrey','darkseagreen', 'silver', 'cyan','skyblue', 'purple','pink','orange']
    cv = []
    for d in dist_t:
        if d == dist_v[0]: cv.append(cm[0])
        if d == dist_v[1]: cv.append(cm[1])
        if d == dist_v[2]: cv.append(cm[2])
        if d == dist_v[3]: cv.append(cm[3])
        if d == dist_v[4]: cv.append(cm[4])
        if d == dist_v[5]: cv.append(cm[5])
        if d == dist_v[6]: cv.append(cm[6])
        if d == dist_v[7]: cv.append(cm[7])
        if d == dist_v[8]: cv.append(cm[8])
    
        
    fig = plt.figure('x.png')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=cv, s = 2200, edgecolor= 'k')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    ax2 = plt.gca(projection='3d') 
    ax2._axis3don = False
    
    
    ax.view_init(30, 90)
    plt.savefig('Pd_bulk_structure.png', dpi = dpi)
    
if __name__ == "__main__":
    plot_NN()
else: pass
