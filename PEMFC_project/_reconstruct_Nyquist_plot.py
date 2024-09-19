import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns

#__all__ =  ['get_points', 'get_points_c','KDE_generate_params']


def get_points(upper, lowest, y_pred, std, zf_samples, param_samples):
    get_list = [ i for i in range(y_pred.shape[0]) if y_pred[i]>lowest and y_pred[i]<=upper and std[i]<=0.05*(upper+lowest)]
    get_y = y_pred[get_list].ravel()
    get_std = std[get_list].ravel()
    #get_std = get_std[np.where(get_std<=25)].ravel()
    get_zf = zf_samples[get_list,:]
    get_param_sam = param_samples[get_list,:]

    return get_y, get_std, get_zf, get_param_sam


def get_points_c(c, y_pred,zf_samples, param_samples):
    get_list = [ i for i in range(y_pred.shape[0]) if y_pred[i]==c]
    get_y = y_pred[get_list].ravel()
    get_zf = zf_samples[get_list,:]
    get_param_sam = param_samples[get_list,:]

    return get_y, get_zf, get_param_sam


def plot_param_dist(params,y_c,plt_title=None, plot_kde=False) :
    fig, axs = plt.subplots(nrows=2,ncols=4,figsize=(20, 8))
    p=0
    titles = ['L','Rm','Q','phi','Rct','Cdl','Rt','O2 stoi']
    palette = sns.color_palette("Paired",8)
    for i in range(2):
        for j in range(4):
            if p==0 :
                axs[i,j].hist(params[:,p],color=palette[p],histtype= 'stepfilled',edgecolor='k' )
                axs[i,j].set_title(titles[p])
                p = p+1
            elif p<7 :
                sns.histplot(params[:,p],ax=axs[i,j],kde=plot_kde,color=palette[p], element="step", fill=True)
                axs[i,j].set_title(titles[p])
                p = p+1
                continue
            else :
                sns.histplot(y_c,ax=axs[i,j],bins=20,color=palette[p], element="step", fill=True)
                break
            
                
    fig.suptitle(plt_title, fontsize = 'large')


def KDE_samples(data, sample_size):
    
    def optimize_bandwidths(data):
        bandwidths = 10**np.linspace(-3,1,100)
        
        grid = GridSearchCV(KernelDensity(),
                    {'kernel':('gaussian', 'tophat'),
                    'bandwidth': bandwidths},
                   cv=5,n_jobs=-1)
        grid.fit(data[:,None])
        return grid.best_params_['kernel'],grid.best_params_['bandwidth']
    
    kernel,bw = optimize_bandwidths(data)
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(data[:,None])
    new_samples = kde.sample(n_samples=sample_size)
    return new_samples.ravel()

def KDE_generate_params(sam, numbers):
    samples = np.zeros((numbers,sam.shape[1]))
    for i in range(sam.shape[1]):
        param_new = KDE_samples(sam[:,i], numbers)
        #param_new = np.abs(param_new)
        samples[:,i] = param_new
        
    return samples
