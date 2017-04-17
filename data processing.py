# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:38:41 2017

@author: Musama
"""

import pandas as pd
import numpy as np

# 读取数据
data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'

# 因子数据
PE_ttm =  pd.read_csv(data_dir + 'PE_ttm.csv',header = None).values
PS_ttm =  pd.read_csv(data_dir + 'PS_ttm.csv',header = None).values
DaysCount_UnderMA =  pd.read_csv(data_dir + 'DaysCount_UnderMA.csv',header = None).values
EPfwd12 =  pd.read_csv(data_dir + 'EPfwd12.csv',header = None).values
HeteroRsquare_FF_d20 =  pd.read_csv(data_dir + 'HeteroRsquare_FF_d20.csv',header = None).values
PreRet_M_IndusRet_Citic1_m1 =  pd.read_csv(data_dir + 'PreRet_M_IndusRet_Citic1_m1.csv',header = None).values
Reversal_DongFang =  pd.read_csv(data_dir + 'Reversal_DongFang.csv',header = None).values
Seasonal_NetProfitGrowth_YOY =  pd.read_csv(data_dir + 'Seasonal_NetProfitGrowth_YOY.csv',header = None).values
Seasonal_OperatingRevenueGrowth_YOY =  pd.read_csv(data_dir + 'Seasonal_OperatingRevenueGrowth_YOY.csv',header = None).values
Seasonal_ROA =  pd.read_csv(data_dir + 'Seasonal_ROA.csv',header = None).values
Seasonal_ROE =  pd.read_csv(data_dir + 'Seasonal_ROE.csv',header = None).values
TotalValue =  pd.read_csv(data_dir + 'TotalValue.csv',header = None).values
Turnover_ols_FloatValue_d20 =  pd.read_csv(data_dir + 'Turnover_ols_FloatValue_d20.csv',header = None).values
Turnover_d5_D_d60 =  pd.read_csv(data_dir + 'Turnover_d5_D_d60.csv',header = None).values
DRet_max_d20 =  pd.read_csv(data_dir + 'DRet_max_d20.csv',header = None).values
Turnover =  pd.read_csv(data_dir + 'Turnover.csv',header = None).values


# 处理并存储至本地
labeldata_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\LabelData\\'
def labeler(factorname,group = 10):
    beginidx = 60
    exec('tmpfactor =' + factorname) 
    SynFactor = np.zeros(tmpfactor.shape)
    for t in np.arange(beginidx,tmpfactor.shape[0]):
        tmp_X = tmpfactor[t,:]
        tmpvar = np.vstack([np.arange(0,len(tmp_X)),tmp_X]).T
        tmpvar = pd.DataFrame(tmpvar,columns = ['id','scores'])
        tmpvar = tmpvar.sort_values(by = 'scores')
        n = len(tmpvar.scores)- tmpvar.scores.isnull().sum()
        N = np.floor(n/group)
        N = N.astype(np.int)
        tmpvar1 = np.repeat(np.arange(group),N)
        tmpvar2 = np.repeat(np.nan,len(tmpvar.scores)-len(tmpvar1))
        tmpvar1 = np.concatenate([tmpvar1,tmpvar2])
        tmpvar['rank'] = tmpvar1
        score1 = tmpvar.sort_values(by = 'id')['rank']
        SynFactor[t,:] = score1
    SynFactor  = pd.DataFrame(SynFactor)
    SynFactor.to_csv(labeldata_dir+factorname+'.csv',index = False)
    
    
factorlist = ['PE_ttm','PS_ttm','EPfwd12','Seasonal_NetProfitGrowth_YOY','Seasonal_ROA',\
              'Seasonal_ROE','Seasonal_OperatingRevenueGrowth_YOY','TotalValue',\
              'DRet_max_d20','Turnover','Turnover_d5_D_d60','Turnover_ols_FloatValue_d20','DaysCount_UnderMA','HeteroRsquare_FF_d20','PreRet_M_IndusRet_Citic1_m1','Reversal_DongFang','EPfwd12']
for f in factorlist:
    labeler(f,group = 10)
    




# 排序处理并存储至本地
rankdata_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\RankData\\'
def ranker(factorname):
    beginidx = 60
    exec('tmpfactor =' + factorname) 
    SynFactor = np.zeros(tmpfactor.shape)
    for t in np.arange(beginidx,tmpfactor.shape[0]):
        tmp_X = tmpfactor[t,:]
        score1 = pd.Series(tmp_X).rank().values
        SynFactor[t,:] = score1
    SynFactor  = pd.DataFrame(SynFactor)
    SynFactor.to_csv(rankdata_dir+factorname+'.csv',index = False)
    
    
factorlist = ['PE_ttm','PS_ttm','EPfwd12','Seasonal_NetProfitGrowth_YOY','Seasonal_ROA',\
              'Seasonal_ROE','Seasonal_OperatingRevenueGrowth_YOY','TotalValue',\
              'DRet_max_d20','Turnover','Turnover_d5_D_d60','Turnover_ols_FloatValue_d20','DaysCount_UnderMA ','HeteroRsquare_FF_d20','PreRet_M_IndusRet_Citic1_m1','Reversal_DongFang','EPfwd12']
for f in factorlist:
    ranker(f)

    
# 处理因变量
data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
ClosePrice =  pd.read_csv(data_dir + 'ClosePrice.csv',header = None)
Ret_m = ClosePrice.shift(-20)/ClosePrice-1
Ret_m.to_csv('C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\Ret_m.csv',index = False)
## 收益率分组
labeldata_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\LabelData\\'
def ret_label(percentage = 0.3):
    beginidx = 60
    tmpfactor = Ret_m.values
    SynFactor = np.zeros(tmpfactor.shape)
    for t in np.arange(beginidx,tmpfactor.shape[0]):
        tmp_X = -tmpfactor[t,:]
        tmpvar = np.vstack([np.arange(0,len(tmp_X)),tmp_X]).T
        tmpvar = pd.DataFrame(tmpvar,columns = ['id','scores'])
        tmpvar = tmpvar.sort_values(by = 'scores')
        n = len(tmpvar.scores)- tmpvar.scores.isnull().sum()
        N = np.floor(n*(percentage))
        N = N.astype(np.int)
        tmpvar['rank'] = np.nan
        tmpvar['rank'].values[:n][:N] = 1
        tmpvar['rank'].values[:n][-N:] = 0
        score1 = tmpvar.sort_values(by = 'id')['rank']
        SynFactor[t,:] = score1
    SynFactor  = pd.DataFrame(SynFactor)
    SynFactor.to_csv(labeldata_dir+'Ret_m.csv',index = False)
    
 
