# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:05:09 2017

@author: Musama
"""

import pandas as pd
import numpy as np

# 读取数据
data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
HighPrice =  pd.read_csv(data_dir + 'HighPrice.csv',header = None).values
LowPrice =  pd.read_csv(data_dir + 'LowPrice.csv',header = None).values
ListDays =  pd.read_csv(data_dir + 'ListDays.csv',header = None).values
SpecialTreat =  pd.read_csv(data_dir + 'SpecialTreat.csv',header = None).values
PreRet_M_IndusRet_Citic2_m3 =  pd.read_csv(data_dir + 'PreRet_M_IndusRet_Citic2_m3.csv',header = None).values
PE_ttm =  pd.read_csv(data_dir + 'PE_ttm.csv',header = None).values
DRet_max_d20 =  pd.read_csv(data_dir + 'DRet_max_d20.csv',header = None).values
Turnover_corr_d10_Price =  pd.read_csv(data_dir + 'Turnover_corr_d10_Price.csv',header = None).values
PreRet_M_IndusRet_Citic2_m1 =  pd.read_csv(data_dir + 'PreRet_M_IndusRet_Citic2_m1.csv',header = None).values
HeteroRsquare_CAPM_d20 =  pd.read_csv(data_dir + 'HeteroRsquare_CAPM_d20.csv',header = None).values
TotalValue =  pd.read_csv(data_dir + 'TotalValue.csv',header = None).values
Turnover =  pd.read_csv(data_dir + 'Turnover.csv',header = None).values


class Factor_Model():
     
    def __init__(self,holdperiod,stocknum,feerate,begdate,enddate):
        self.holdperiod = holdperiod
        self.stocknum = stocknum
        self.feerate = feerate
        self.begdate = begdate
        self.enddate = enddate
        self.N_date = 3192        
        self.N_stock = 3193

        
    def rank_scorer(self,factor_weight=0.5):
        beginidx = 60
        SynFactor = np.zeros([self.N_date,self.N_stock])
        data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
        PreRet_M_IndusRet_Citic2_m3 =  pd.read_csv(data_dir + 'PreRet_M_IndusRet_Citic2_m3.csv',header = None).values
        PE_ttm =  pd.read_csv(data_dir + 'PE_ttm.csv',header = None).values

        for t in np.arange(beginidx,self.N_date):
            #因子权重，对因子序列加权
            factor1 = -PreRet_M_IndusRet_Citic2_m3[t,:]       #PreRet_M_IndusRet_Citic2_m3因子
            factor2 = -1/PE_ttm[t,:]     #PE_ttm因子
            # 原数据加权排序打分
            tmp_X = factor_weight * factor1 + (1-factor_weight)* factor2
            tmpvar1 = np.vstack([np.arange(0,len(tmp_X)),tmp_X]).T
            tmpvar1 = pd.DataFrame(tmpvar1,columns = ['id','scores'])
            tmpvar1 = tmpvar1.sort_values(by = 'scores')
            tmpvar1['rank'] = np.arange(len(tmpvar1.scores))
            score1 = tmpvar1.sort_values(by = 'id')['rank']
            score1[np.isnan(tmp_X)] = (np.isnan(tmp_X) == False).sum()+1
            SynFactor[t,:] = score1
        self.SynFactor = SynFactor
        return self
        

    
    def backtest(self):
        
        # 读取数据
        data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
        HighPrice =  pd.read_csv(data_dir + 'HighPrice.csv',header = None).values
        LowPrice =  pd.read_csv(data_dir + 'LowPrice.csv',header = None).values
        ListDays =  pd.read_csv(data_dir + 'ListDays.csv',header = None).values
        SpecialTreat =  pd.read_csv(data_dir + 'SpecialTreat.csv',header = None).values
        index500 =  pd.read_csv(data_dir + 'index500.csv',header = None)
        DRet = pd.read_csv(data_dir + 'DRet.csv',header = None).values
        ret500 = (index500/index500.shift(1)-1).values 
        DRet[np.isnan(DRet)] = 0
        
        begid = 1000
        ret_strategy = np.zeros(self.N_date)
        tradetimes = np.ceil((self.N_date-begid)/self.holdperiod)
        stock_group = list()
        weight_group = list()
#        SynFactor = self.SynFactor
        #-------------------全市场等权配置----------------
        for idx in np.arange(tradetimes,dtype = np.int):
            buyid = begid+idx*self.holdperiod+1
#            tmpindicator = SynFactor[buyid-1,:]
            invalidindex = np.vstack([HighPrice[buyid,:]==LowPrice[buyid,:] , np.isnan(HighPrice[buyid,:]) , SpecialTreat[buyid-1,:]==1 , ListDays[buyid-1,:]<60]).any(axis = 0)        #invalid的股票，向量
            tmpindicator = -RF_predictor(buyid,period = 12, freq=20)
            valid_idx = invalidindex==False            # 有效的股票
            tmpvar2 = np.vstack([np.arange(len(tmpindicator)),tmpindicator]).T
            tmpvar2 = tmpvar2[valid_idx,:]
        
            tmpvar2 = pd.DataFrame(tmpvar2,columns = ['id','scores'])
            tmpvar2 = tmpvar2.sort_values(by = 'scores')['id'].values     # 提取排序后的股票id            
            stock_id = tmpvar2[np.arange(self.stocknum)]
            stock_id = stock_id.astype(np.int)                       
                   
            # 计算weight
            stock_weight = 1/self.stocknum * np.ones(self.stocknum)      # 等权权重，列向量
            stock_group.append(stock_id)        # 当期全市场选股id
            weight_group.append(stock_weight)          # 当期全市场选股权重     
            
            # 计算组合收益率
            if idx<tradetimes:
                tmpret = DRet[(buyid+1):(buyid+self.holdperiod+1),stock_id].dot(stock_weight)      #本次买入后持仓组合收益率
                ret_strategy[(buyid+1):(buyid+self.holdperiod+1)] = tmpret
            else:
                tmpret = DRet[(buyid+1):,stock_id].dot(stock_weight)
                ret_strategy[(buyid+1):] = tmpret
            print('Trade Times:%.i completed'%(idx+1))
            
        ret500[:(begid+2)] = 0
        ret_strategy[np.isnan(ret_strategy)] = 0
        ret500[np.isnan(ret500)] = 0                                  
        ret_hedge500 = ret_strategy - ret500[:,0]
        nav_long = np.cumprod(1+ret_strategy)
        nav_hedge500 = np.cumprod(1+ret_hedge500)
        #返回值
        self.stcok_group = stock_group
        self.weight_group = weight_group
        self.ret_strategy = ret_strategy
        self.ret_hedge500 = ret_hedge500
        self.nav_hedge500 = nav_hedge500
        self.nav_long = nav_long
        self.return_long = self.nav_long[-1]**(245/(self.N_date-begid))-1
        self.return_hedge500 = self.nav_hedge500[-1]**(245/(self.N_date-begid))-1
        self.volatility_hedge500 = np.std(self.ret_hedge500)*np.sqrt(245)        
        self.IR_hedge500 = self.return_hedge500/self.volatility_hedge500
        return self
        
        
    def performance_report(self):
        drawdown_long = [self.nav_long[i]/np.max(self.nav_long[:(i+1)])-1 for i in np.arange(self.N_date)]
        drawdown_hedge500 = [self.nav_hedge500[i]/np.max(self.nav_hedge500[:(i+1)])-1 for i in np.arange(self.N_date)]                
        max_drawdown_end = np.where(drawdown_hedge500==np.min(drawdown_hedge500))[0][0]
        max_drawdown_beg = np.where(self.nav_hedge500[:(max_drawdown_end+1)]==np.max(self.nav_hedge500[:(max_drawdown_end+1)]))[0][0]
        self.drawdown_long = drawdown_long
        self.draw_hedge500 = drawdown_hedge500
        self.max_drawdown_hedge500 = np.min(drawdown_hedge500)
        self.max_drawdown_hedge500_period = [max_drawdown_beg,max_drawdown_end]
        
        print('-------------------------------------------------------------------------')
        print('Long position annualized return: %.4f'%(self.return_long))
        print('Hedging HS500 annualized return: %.4f'%(self.return_hedge500))
        print('Hedging HS500 Maxmium Drawdown: %.3f, Drawdown period: from %i, to %i'\
              %(self.max_drawdown_hedge500,max_drawdown_beg,max_drawdown_end))
        print('-------------------------------------------------------------------------')

        return self

def RF_predictor(buyid,period = 12, freq=20):
    import numpy as np
    import pandas as pd
    
    # 构造自变量
    data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
    idx = np.linspace(buyid-1-freq*period,buyid-1-freq,period,dtype = np.int)
    ClosePrice = pd.read_csv(data_dir + 'ClosePrice.csv',header = None).values
    
    # 预测期输入变量
    factor1_pre = PreRet_M_IndusRet_Citic2_m3[buyid-1,:]
    factor2_pre = PE_ttm[buyid-1,:]
    factor3_pre = DRet_max_d20[buyid-1,:]
    factor4_pre = Turnover_corr_d10_Price[buyid-1,:]   
    factor5_pre = PreRet_M_IndusRet_Citic2_m1[buyid-1,:]    
    factor6_pre = HeteroRsquare_CAPM_d20 [buyid-1,:]
    factor7_pre = TotalValue[buyid-1,:]
    factor8_pre = Turnover[buyid-1,:]
    X_pre = np.vstack([factor1_pre,factor2_pre,factor3_pre,factor4_pre,factor5_pre,factor6_pre,factor7_pre,factor8_pre]).T
    invalidindex = np.vstack([np.isnan(X_pre).any(axis=1),HighPrice[buyid,:]==LowPrice[buyid,:] , np.isnan(HighPrice[buyid,:]) , SpecialTreat[buyid-1,:]==1 , ListDays[buyid-1,:]<60]).any(axis = 0)        #invalid的股票，向量
    valid_idx = invalidindex ==False    
    X_pre = X_pre[valid_idx,:]

    factor1 = np.concatenate(PreRet_M_IndusRet_Citic2_m3[idx,:])
    factor2 = np.concatenate(PE_ttm[idx,:])
    factor3 = np.concatenate(DRet_max_d20[idx,:])
    factor4 = np.concatenate(Turnover_corr_d10_Price[idx,:])
    factor5 = np.concatenate(PreRet_M_IndusRet_Citic2_m1[idx,:])
    factor6 = np.concatenate(HeteroRsquare_CAPM_d20[idx,:])
    factor7 = np.concatenate(TotalValue[idx,:])
    factor8 = np.concatenate(Turnover[idx,:])
    X = np.vstack([factor1,factor2,factor3,factor4,factor5,factor6,factor7,factor8]).T
    
    # 构造因变量
    Ret_m = ClosePrice[20:,:]/ClosePrice[:-20,:]-1
    tmp = np.zeros([20,3193])
    tmp[:] = np.nan
    Ret_m = np.vstack([Ret_m,tmp])
    y = np.concatenate(Ret_m[idx,:])
    df = pd.DataFrame(np.column_stack([X,y]))
    df = df.dropna()
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    #构建分类器
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_pre_std = stdsc.fit_transform(X_pre)
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200,max_features=8,n_jobs=-1)
    rf = rf.fit(X_train_std,y_train)
    ret_pre = rf.predict(X_pre_std)
    result = np.zeros(len(valid_idx))
    result[:] =np.nan
    result[valid_idx] = ret_pre
    return result


f1 = Factor_Model(holdperiod = 20,stocknum = 100,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.backtest()
f1.performance_report()

