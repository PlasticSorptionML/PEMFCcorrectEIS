from _equivalent_circuits import *
from _data_summery import *
from _reconstruct_Nyquist_plot import *
import pandas as pd
import os

##path = os.path.join('C:\\Users\\mero_\\Desktop\\y2s2\\EIS_pile_up\\testing\\')
###eis_ny = pd.read_csv(path+'EIS_O2_STOI_ny.csv',header=0,index_col=0)
##o2_ny = pd.read_csv(path+'O2_STOI_ny.csv',header=0,index_col=0)
##
##df_count = target_summery(o2_ny)
get_points_c(1.8, y_c_pred,zf_samples, samples_2)
