# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:05:09 2017

@author: Musama
"""

import pandas as pd
import numpy as np

# 读取原始数据
data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
ClosePrice =  pd.read_csv(data_dir + 'ClosePrice.csv',header = None).values
HighPrice =  pd.read_csv(data_dir + 'HighPrice.csv',header = None).values
LowPrice =  pd.read_csv(data_dir + 'LowPrice.csv',header = None).values
ListDays =  pd.read_csv(data_dir + 'ListDays.csv',header = None).values
SpecialTreat =  pd.read_csv(data_dir + 'SpecialTreat.csv',header = None).values
index300 =  pd.read_csv(data_dir + 'index300.csv',header = None).values
index500 =  pd.read_csv(data_dir + 'index500.csv',header = None).values

# 因子数据

# 自变量因子排序后分为五组,因变量分为1，0，1为高收益组
labeldata_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\labelData\\'
Ret_m = pd.read_csv(labeldata_dir + 'Ret_m.csv').values

##导入自变量
labeldata_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
TotalValue =  pd.read_csv(labeldata_dir + 'TotalValue.csv',header = None).values
PB_lf =  pd.read_csv(labeldata_dir + 'PB_lf.csv',header = None).values
HeteroRsquare_FF_d20 =  pd.read_csv(labeldata_dir + 'HeteroRsquare_FF_d20.csv',header = None).values
PreRet_M_IndusRet_Citic1_m1 =  pd.read_csv(labeldata_dir + 'PreRet_M_IndusRet_Citic1_m1.csv',header = None).values
PreRet_M_IndusRet_Citic1_m3 =  pd.read_csv(labeldata_dir + 'PreRet_M_IndusRet_Citic1_m3.csv',header = None).values
Seasonal_NetProfitGrowth_YOY =  pd.read_csv(labeldata_dir + 'Seasonal_NetProfitGrowth_YOY.csv',header = None).values
Turnover_ols_TotalValue_d20 =  pd.read_csv(labeldata_dir + 'Turnover_ols_TotalValue_d20.csv',header = None).values
Reversal_DongFang =  pd.read_csv(labeldata_dir + 'Reversal_DongFang.csv',header = None).values

class Factor_Model():
     
    def __init__(self,holdperiod,stocknum,feerate,begdate,enddate):
        self.holdperiod = holdperiod
        self.stocknum = stocknum
        self.feerate = feerate
        self.begdate = begdate
        self.enddate = enddate
        self.N_date = 3217        
        self.N_stock = 3242

    def rank_scorer(self,factor_weight=0.5):
        beginidx = 60
        SynFactor = np.zeros([self.N_date,self.N_stock])
        for t in np.arange(beginidx,self.N_date):
            #因子权重，对因子序列加权
            factor1 = PB_lf[t,:]
            factor2 = PB_lf[t,:]  
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
    
            
    def scorer_backtest(self):  
        # 读取数据
        data_dir = 'C:\\Users\\Musama\\Desktop\\Factor Models\\Data\\'
        HighPrice =  pd.read_csv(data_dir + 'HighPrice.csv',header = None).values
        LowPrice =  pd.read_csv(data_dir + 'LowPrice.csv',header = None).values
        ListDays =  pd.read_csv(data_dir + 'ListDays.csv',header = None).values
        SpecialTreat =  pd.read_csv(data_dir + 'SpecialTreat.csv',header = None).values
        index300 =  pd.read_csv(data_dir + 'index300.csv',header = None)
        index500 =  pd.read_csv(data_dir + 'index500.csv',header = None)
        DRet = pd.read_csv(data_dir + 'DRet.csv',header = None).values
        ret300 = (index300/index300.shift(1)-1).values 
        ret500 = (index500/index500.shift(1)-1).values 
        DRet[np.isnan(DRet)] = 0
        
        begid = 1214 
        ret_strategy = np.zeros(self.N_date)
        ret_short = np.zeros(self.N_date)
        tradetimes = np.ceil((self.N_date-begid)/self.holdperiod)
        stock_group = list()
        weight_group = list()
        predict_score = list()
#        SynFactor = self.SynFactor
        
        #-------------------全市场等权配置----------------
        for idx in np.arange(tradetimes,dtype = np.int):
            buyid = begid+idx*self.holdperiod+1
#            tmpindicator = SynFactor[buyid-1,:]
            result = RF_predictor(buyid,method=method)
            tmpindicator = -result[0]           #提取预测打分
            predict_score.append(result[1])     #存储预测accuracy
            invalidindex = np.vstack([HighPrice[buyid,:]==LowPrice[buyid,:] , np.isnan(HighPrice[buyid,:]) , SpecialTreat[buyid-1,:]==1 , ListDays[buyid-1,:]<60,np.isnan(tmpindicator)]).any(axis = 0)        #invalid的股票，向量
            valid_idx = invalidindex==False            # 有效的股票
            tmpvar2 = np.vstack([np.arange(len(tmpindicator)),tmpindicator]).T
            tmpvar2 = tmpvar2[valid_idx,:]
            tmpvar2 = pd.DataFrame(tmpvar2,columns = ['id','scores'])
            tmpvar2 = tmpvar2.sort_values(by = 'scores')['id'].values     # 提取排序后的股票id            
            stock_id = tmpvar2[np.arange(self.stocknum)]
            stock_id = stock_id.astype(np.int)                       
            short_stock_id = tmpvar2[-self.stocknum:]
            short_stock_id = short_stock_id.astype(np.int)                       

            # 计算weight
            stock_weight = 1/self.stocknum * np.ones(self.stocknum)      # 等权权重，列向量
            short_stock_weight = 1/self.stocknum * np.ones(self.stocknum)      # 等权权重，列向量
            stock_group.append(stock_id)        # 当期全市场选股id
            weight_group.append(stock_weight)          # 当期全市场选股权重     
            
            # 计算组合收益率
            if idx<tradetimes:
                tmpret = DRet[(buyid+1):(buyid+self.holdperiod+1),stock_id].dot(stock_weight)      #本次买入后持仓组合收益率
                ret_strategy[(buyid+1):(buyid+self.holdperiod+1)] = tmpret
                tmpret = DRet[(buyid+1):(buyid+self.holdperiod+1),short_stock_id].dot(short_stock_weight)      #本次买入后持仓组合收益率
                ret_short[(buyid+1):(buyid+self.holdperiod+1)] = tmpret
            else:
                tmpret = DRet[(buyid+1):,stock_id].dot(stock_weight)
                ret_strategy[(buyid+1):] = tmpret
                tmpret = DRet[(buyid+1):,short_stock_id].dot(short_stock_weight)
                ret_short[(buyid+1):] = tmpret
            print('Trade Times:%.i completed'%(idx+1))
            
        ret300[:(begid+2)] = 0
        ret_strategy[np.isnan(ret_strategy)] = 0
        ret_short[np.isnan(ret_short)] = 0
        ret_longshort = ret_strategy - ret_short
        ret300[np.isnan(ret500)] = 0                                  
        ret_hedge300 = ret_strategy - ret300[:,0]
        nav_long = np.cumprod(1+ret_strategy)
        nav_longshort = np.cumprod(1+ret_longshort)
        nav_hedge300 = np.cumprod(1+ret_hedge300)            
        ret500[:(begid+2)] = 0
        ret500[np.isnan(ret500)] = 0                                  
        ret_hedge500 = ret_strategy - ret500[:,0]
        nav_hedge500 = np.cumprod(1+ret_hedge500)
        #返回值
        self.stcok_group = stock_group
        self.weight_group = weight_group
        self.pre_score = predict_score
        # return
        self.ret_strategy = ret_strategy
        self.ret_lonngshort= ret_longshort
        self.ret_hedge300 = ret_hedge300
        self.ret_hedge500 = ret_hedge500
        
        # nav
        self.nav_long = nav_long
        self.nav_longshort = nav_longshort
        self.nav_hedge300 = nav_hedge300        
        self.nav_hedge500 = nav_hedge500
        
        # annulised return
        self.return_long = self.nav_long[-1]**(245/(self.N_date-begid))-1
        self.return_longshort = self.nav_longshort[-1]**(245/(self.N_date-begid))-1
        self.return_hedge300 = self.nav_hedge300[-1]**(245/(self.N_date-begid))-1
        self.volatility_hedge300 = np.std(self.ret_hedge300)*np.sqrt(245)        
        self.IR_hedge300 = self.return_hedge300/self.volatility_hedge300
        
        self.return_hedge500 = self.nav_hedge500[-1]**(245/(self.N_date-begid))-1
        self.volatility_hedge500 = np.std(self.ret_hedge500)*np.sqrt(245)        
        self.IR_hedge500 = self.return_hedge500/self.volatility_hedge500
        return self      
        
    def performance_report(self):
        
        # drawdown
        drawdown_long = [self.nav_long[i]/np.max(self.nav_long[:(i+1)])-1 for i in np.arange(self.N_date)]
        drawdown_longshort = [self.nav_longshort[i]/np.max(self.nav_longshort[:(i+1)])-1 for i in np.arange(self.N_date)]
        max_drawdownlongshort_end = np.where(drawdown_longshort==np.min(drawdown_longshort))[0][0]
        max_drawdownlongshort_beg = np.where(self.nav_hedge300[:(max_drawdownlongshort_end+1)]==np.max(self.nav_hedge300[:(max_drawdownlongshort_end+1)]))[0][0]

        # hedge 300
        drawdown_hedge300 = [self.nav_hedge300[i]/np.max(self.nav_hedge300[:(i+1)])-1 for i in np.arange(self.N_date)]                
        max_drawdown300_end = np.where(drawdown_hedge300==np.min(drawdown_hedge300))[0][0]
        max_drawdown300_beg = np.where(self.nav_hedge300[:(max_drawdown300_end+1)]==np.max(self.nav_hedge300[:(max_drawdown300_end+1)]))[0][0]

        # hedge 300                 
        drawdown_hedge500 = [self.nav_hedge500[i]/np.max(self.nav_hedge500[:(i+1)])-1 for i in np.arange(self.N_date)]                
        max_drawdown500_end = np.where(drawdown_hedge500==np.min(drawdown_hedge500))[0][0]
        max_drawdown500_beg = np.where(self.nav_hedge500[:(max_drawdown500_end+1)]==np.max(self.nav_hedge500[:(max_drawdown500_end+1)]))[0][0]
        
        self.drawdown_long = drawdown_long
        self.drawdown_longshort = drawdown_longshort
        self.max_drawdown_longshort= np.min(drawdown_longshort)
        self.max_drawdown_longshort_period = [max_drawdownlongshort_beg,max_drawdownlongshort_end]
        
        self.draw_hedge300 = drawdown_hedge300
        self.max_drawdown_hedge300 = np.min(drawdown_hedge300)
        self.max_drawdown_hedge300_period = [max_drawdown300_beg,max_drawdown300_end]

        self.draw_hedge500 = drawdown_hedge500
        self.max_drawdown_hedge500 = np.min(drawdown_hedge500)
        self.max_drawdown_hedge500_period = [max_drawdown500_beg,max_drawdown500_end]
        df = pd.DataFrame([self.nav_long,self.nav_longshort,self.nav_hedge300,self.nav_hedge500]).T
        df.columns = ['Long Postition','Longshort Hedging','Hedging HS300','Hedging ZZ500']
        df.plot(figsize = [12,8],grid=1)
        print('-------------------------------------------------------------------------')
        print('Long position annualized return: %.4f'%(self.return_long))
        print('Long Short annualized return: %.4f'%(self.return_longshort))
        print('Long Short Maxmium Drawdown: %.3f, Drawdown period: from %i, to %i'\
              %(self.max_drawdown_longshort,max_drawdownlongshort_beg,max_drawdownlongshort_end))
        print('Hedging HS300 annualized return: %.4f'%(self.return_hedge300))
        print('Hedging HS300 Maxmium Drawdown: %.3f, Drawdown period: from %i, to %i'\
              %(self.max_drawdown_hedge300,max_drawdown300_beg,max_drawdown300_end))
        print('Hedging HS500 annualized return: %.4f'%(self.return_hedge500))
        print('Hedging HS500 Maxmium Drawdown: %.3f, Drawdown period: from %i, to %i'\
              %(self.max_drawdown_hedge500,max_drawdown500_beg,max_drawdown500_end))
        print('-------------------------------------------------------------------------')
        return self

        
    
def RF_predictor(buyid,method,period = 12, freq=20):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()    
    # 构造自变量
    idx = np.linspace(buyid-21-freq*period,buyid-21,period,dtype = np.int)    
    # 预测期输入变量
    factor1_pre = TotalValue[buyid-1,:]
    factor2_pre = PB_lf[buyid-1,:]
    factor3_pre = Reversal_DongFang[buyid-1,:]
    factor4_pre = HeteroRsquare_FF_d20[buyid-1,:]   
    factor5_pre = PreRet_M_IndusRet_Citic1_m1[buyid-1,:]    
    factor6_pre = PreRet_M_IndusRet_Citic1_m3 [buyid-1,:]
    factor7_pre = Seasonal_NetProfitGrowth_YOY[buyid-1,:]
    factor7_pre = Seasonal_NetProfitGrowth_YOY[buyid-1,:]
    factor8_pre = Turnover_ols_TotalValue_d20[buyid-1,:]



    # 并为预测自变量矩阵
    X_pre = np.vstack([factor1_pre,factor2_pre,factor3_pre,factor4_pre,\
                       factor5_pre,factor6_pre,factor7_pre,factor8_pre]).T

        
    # 剔除停牌、ST
    invalidindex = np.vstack([np.isnan(X_pre).any(axis=1),HighPrice[buyid,:]==LowPrice[buyid,:] , np.isnan(HighPrice[buyid,:]) , SpecialTreat[buyid-1,:]==1 , ListDays[buyid-1,:]<60]).any(axis = 0)        #invalid的股票，向量
    valid_idx = invalidindex ==False        #有效的id
    X_pre = X_pre[valid_idx,:]          #有效的预测自变量
    X_pre = stdsc.fit_transform(X_pre)
    y_true = Ret_m[buyid-1,:][valid_idx]
    # 训练组自变量
    factor1 = np.concatenate(np.log(TotalValue[idx,:]))
    factor2 = np.concatenate(PB_lf[idx,:])
    factor3 = np.concatenate(Reversal_DongFang[idx,:])
    factor4 = np.concatenate(HeteroRsquare_FF_d20[idx,:])   
    factor5 = np.concatenate(PreRet_M_IndusRet_Citic1_m1[idx,:])    
    factor6 = np.concatenate(PreRet_M_IndusRet_Citic1_m3[idx,:])
    factor7 = np.concatenate(Seasonal_NetProfitGrowth_YOY[idx,:])
    factor8 = np.concatenate(Turnover_ols_TotalValue_d20[idx,:])

    X = np.vstack([factor1,factor2,factor3,factor4,factor5,\
                       factor6,factor7,factor8]).T
                   
    y = np.concatenate(Ret_m[idx,:])
    df = pd.DataFrame(np.column_stack([X,y]))
    df = df.dropna()
    X_train = df.iloc[:,:-1].values    
    X_train = stdsc.fit_transform(X_train)
    y_train = df.iloc[:,-1].values

    from sklearn.metrics import accuracy_score
    
    if method=='LinearRegression':
        #Linear Regression回归
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols = ols.fit(X_train,y_train)
        ret_pre = ols.predict(X_pre)
        y_pre = np.where(ret_pre>0.5,1,0)
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])
    elif method=='LogisticRegression':
        #Logistic分类器
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr = lr.fit(X_train,y_train)
        y_pre = lr.predict(X_pre)
        ret_pre = lr.predict_proba(X_pre)[:,1]
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])
    elif method=='RandomForest':
        #Random Forest分类器
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300,max_features='auto',n_jobs=-1)
        rf = rf.fit(X_train,y_train)
        y_pre = rf.predict(X_pre)
        ret_pre = rf.predict_proba(X_pre)[:,1]
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])
    elif method=='RandomForest_Regression':
        #Random Forest分类器
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300,max_features='auto',n_jobs=-1)
        rf = rf.fit(X_train,y_train)
        ret_pre = rf.predict(X_pre)
        y_pre = np.where(ret_pre>0.5,1,0)
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])
    elif method=='Adaboost':
        #AdaBoost分类器    
        from sklearn.ensemble import AdaBoostClassifier
        adb = AdaBoostClassifier(n_estimators=300)
        adb = adb.fit(X_train,y_train)
        y_pre = adb.predict(X_pre)
        ret_pre = adb.predict_proba(X_pre)[:,1]
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])
    elif method=='Adaboost_Regression':
        #AdaBoost分类器    
        from sklearn.ensemble import AdaBoostRegressor
        adb = AdaBoostRegressor(n_estimators=300)
        adb = adb.fit(X_train,y_train)
        ret_pre = adb.predict(X_pre)
        y_pre = np.where(ret_pre>0.5,1,0)
        tmp = pd.DataFrame([y_true,y_pre]).T.dropna().values 
        score = accuracy_score(tmp[:,0],tmp[:,1])  
        
        
    result = np.zeros(len(valid_idx))
    result[:] =np.nan
    result[valid_idx] = ret_pre
    return result,score


    
#单因子
#------------------------------------------------------------------------------
f1 = Factor_Model(holdperiod = 20,stocknum = 100,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.rank_scorer()
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('PB_lf.csv')          

        
#------------------------------------------------------------------------------    
method = 'LinearRegression'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'LinearRegression'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')

method = 'LogisticRegression'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'LogisticRegression'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest'
f1 = Factor_Model(holdperiod = 20,stocknum = 100,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_100.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest'
f1 = Factor_Model(holdperiod = 20,stocknum = 300,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_300.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'Adaboost'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')

method = 'Adaboost'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'RandomForest_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 300,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_300.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'Adaboost_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 150,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_150.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'Adaboost_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 200,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_200.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')


method = 'Adaboost_Regression'
f1 = Factor_Model(holdperiod = 20,stocknum = 300,feerate = 0,begdate = '2009/01/01',enddate ='2017/01/01')
f1.scorer_backtest()
f1.performance_report()
ret = pd.DataFrame([f1.ret_lonngshort,f1.ret_hedge300,f1.ret_hedge500]).T     
ret.to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_300.csv')                                                                                                  
pd.Series(f1.pre_score).to_csv('C:\\Users\\Musama\\Desktop\\'+method+'_score.csv')