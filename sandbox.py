# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 23:06:16 2021

@author: User
"""


import pandas as pd
import numpy as np
from aux_functions import *

from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from sklearn.metrics import make_scorer, precision_score


api_key = ''    #
api_secret = '' #

tf = '1h'
symbol = "BTCUSDT"
btcdataset=get_market_data(symbol,tf,api_key,api_secret)

supports_lookback=1000
supports_rank=3
all_supports=calc_supports_3(btcdataset,supports_lookback,supports_rank,True)            


#tradetype="short"
tradetype="long"
tp=1.024 #1.025
sl=0.96 #
tf_adj=1

emawindowlist = list( (np.array([30,50,75,100,150,200,250,300,350,400,500,600,750,950])/tf_adj).astype(int) )
emaratelists = list( (np.array([1,10,30,50,100,200,300,400,600,800])/tf_adj*1).astype(int) )
qvwmawindowlist = [1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,11000000] 
qvwmaratelists = list( (np.array([3,10,30,50,100,200,300,350,400,500])/tf_adj*1).astype(int) )
qvwma_exp = [3,2,1]


params={ 'delta_to_last_trade': False, # delta between now and last trade
   # 'last_tps': list((np.array([500,1000,2000])/tf_adj).astype(int) ) , # count tps in the past n candles
   # 'last_sls':list((np.array([500,1000,2000])/tf_adj).astype(int) ), # count sls in the past n candles
  
  'order_ratio_to_rolling':[{'type':'high',
                            'window':int(100/tf_adj)},
                            {'type':'low',
                            'window':int(100/tf_adj)},
                            {'type':'high',
                            'window':int(300/tf_adj)},
                            {'type':'low',
                            'window':int(300/tf_adj)},
                            {'type':'high',
                            'window':int(500/tf_adj)},
                            {'type':'low',
                            'window':int(500/tf_adj)},
                            {'type':'high',
                            'window':int(1000/tf_adj)},
                            {'type':'low',
                            'window':int(1000/tf_adj) }],
  
  'price_ratio_to_rolling':[{'type':'high',
                            'window':int(100/tf_adj)},
                            {'type':'low',
                            'window':int(100/tf_adj)},
                            {'type':'high',
                            'window':int(300/tf_adj)},
                            {'type':'low',
                            'window':int(300/tf_adj)},
                            {'type':'high',
                            'window':int(500/tf_adj)},
                            {'type':'low',
                            'window':int(500/tf_adj)},
                            {'type':'high',
                            'window':int(1000/tf_adj)},
                            {'type':'low',
                            'window':int(1000/tf_adj) }],

  'score':True,
  'abs_vol':True,

  'vol_buffers' : list( (np.array([250,450,700])/tf_adj*1).astype(int) ),
   # 'vol_above_order': [1.03,1.07],
   # 'vol_below_order': [0.97,0.93],
   # 'vol_above_price': [1.03,1.07],
   # 'vol_below_price': [0.97,0.93]

}
qvwma = []
for q in qvwma_exp:
    for i in qvwmawindowlist:
        for j in qvwmaratelists:
            qvwma.append({'window':i,
                          'rate':j,
                          'ratio_to_order':True,
                          'ratio_to_price':True,
                          'exp':q} )
params['qvwma'] = qvwma

# svwma = []
# for q in qvwma_exp:
#     for i in emawindowlist:
#         for j in emaratelists:
#             svwma.append({'window':i,
#                           'rate':j,
#                           'ratio_to_order':True,
#                           'ratio_to_price':True,
#                           'exp':q} )
# params['svwma'] = svwma

ema = []
for i in emawindowlist:
    for j in emaratelists:
        ema.append({'window':i,
                      'rate':j,
                      'ratio_to_order':True,
                      'ratio_to_price':True
                       })
params['ema'] = ema

wma = []
for i in emawindowlist: 
    for j in emaratelists:
        wma.append({'window':i,
                      'rate':j,
                      'ratio_to_order':True,
                      'ratio_to_price':True
                       })
params['wma'] = wma

minmax_hl = []
for i in emawindowlist: 
    for j in emaratelists:
        minmax_hl.append({'window':i,
                      'rate':j,
                      'ratio_to_order':True,
                      'ratio_to_price':True
                       })
params['minmax_hl'] = minmax_hl

rf_df,y,trades, paramsused = process_strategy(all_supports, btcdataset, tp, sl,tradetype, 1000, False, params)


mass= {'ema':[{'window' :200}], 'params': symbol+'_'+tf+'_'+'0_'+'0' }

rf_df,y,trades, paramsused = process_strategy( mass, btcdataset, tp, sl,tradetype, 1000, False, params)




# name = ''
# if name == '':
#     paramstr = all_supports['params'].values[0]
#     import pickle
#     name = 'bundle' + '_' +paramstr + '_' + str(tp)  + '_' + str(sl)  + '_' + str(tradetype)
# else:
#     pass

# with open(name+'.pkl', 'rb') as file:
#     loaded_bundle = pickle.load(file)

# # Access the loaded model from the dictionary
# trades = loaded_bundle['full_data']
# y = loaded_bundle['class']
# paramsused = loaded_bundle['parameters_used']
# all_supports = loaded_bundle['supports']
# trades_params = loaded_bundle['trades_params']
# window= trades_params['window']
# top_n= trades_params['top_n']
# tradetype= trades_params['tradetype']
# tp, sl = trades_params['tp'],trades_params['sl']

# all_supports=calc_supports_3(btcdataset,window,top_n,True)            

# rf_df = loaded_bundle['useful_data']

##############################

(pd.Series(y)).value_counts()

rf_df = rf_df.loc[ (pd.Series(y).isin(['tp','sl'])).values , ]

trades=trades.loc[ (pd.Series(y).isin(['tp','sl'])).values , ]

y=y[ (pd.Series(y).isin(['tp','sl'])).values , ]
y = np.where(y == 'tp', 1, 0)



##############################
ttratio=0.3

features= list(rf_df.columns.values)

nfeats = 6

md=3
sensibility=0.3
nestimators=12
featspertree=1
lr=0.3
subsample=1

import random
loopl=100000

trainlist=[None] * loopl
testlist=[None] * loopl
ntradeslist=[None] * loopl
ntradestrainlist=[None] * loopl
featslist=[None] * loopl
samplesintrain=[None] * loopl
samplesintest=[None] * loopl
tp_ratio=[None] * loopl
test_profits=[None] * loopl
train_profits=[None] * loopl

store_xgboost_params={'test_size':ttratio,
'nfeats':nfeats,
'max_depth':md,
'scale_pos_weight': sensibility,
'n_estimators': nestimators,
'colsample_bytree':featspertree,
'learning_rate':lr,
'subsample':subsample}

feats=[]
i=0
while i < loopl:
    
        random.shuffle(features)
        feats = features[:nfeats]

        X= rf_df.loc[ :,feats].values
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ttratio, random_state=42, shuffle=False)
        train_n_samples= len(y_train)
        train_n_samples_tp= len(y_train[y_train== 1]) /len(y_train)
        test_n_samples= len(y_test)

        eval_set = [(X_test, y_test)]
        model = XGBClassifier(scale_pos_weight=sensibility,
                              max_depth=md,
                              n_estimators=nestimators,
                              colsample_bytree=featspertree,
                              learning_rate=lr,
                              subsample=subsample)
        
        model.fit(X_train, y_train)
        yhat = model.predict(X_train)
        confusion = confusion_matrix(y_train,yhat)
        trainv= confusion[1,1]/sum(confusion[:,1])
        ntraintraddv=confusion[:,1].sum()
        train_profits[i] =  (tp-1)*confusion[1,1] - (1-sl)*confusion[0,1] 

        yhat = model.predict(X_test)
        confusion = confusion_matrix(y_test,yhat)
        testv= confusion[1,1]/sum(confusion[:,1])
    
        ntradv=confusion[:,1].sum()
    
        print( i, " - acc: ", round(trainv,2)," | " , round( testv,2)," | ",ntradv )
        
        trainlist[i] = trainv
        testlist[i] = testv
        ntradeslist[i] = ntradv
        featslist[i] = '"'+ '", "'.join(feats)+'"'
        ntradestrainlist[i] = ntraintraddv
        samplesintrain[i] = train_n_samples
        samplesintest[i] = test_n_samples
        tp_ratio[i] = train_n_samples_tp
        test_profits[i] =  (tp-1)*confusion[1,1] - (1-sl)*confusion[0,1] 
        
        i+=1
        
res_df=pd.DataFrame( {"train_profits":train_profits,
                      "test_profits":test_profits,
                      "samples_in_train":samplesintrain,
                      "samples_in_test":samplesintest,
                      "tp_ratio":tp_ratio,
                      "train_tp_acc":trainlist,
                      "test_tp_acc":testlist,
                      "ntraintrades":ntradestrainlist,
                      "ntrades":ntradeslist,
                      "feats":featslist})

training_test_vline=trades.iloc[ int( (1-ttratio) *len(trades)) ,trades.columns.get_loc("timestamp2") ]

res_df['max_depth']=md
res_df['n_estimators']=nestimators
res_df['scale_pos_weight']=sensibility
res_df['colsample_bytree']=featspertree
res_df['tp']=tp
res_df['sl']=sl
res_df['learning_rate']=lr
res_df['subsample']=subsample
res_df['corte']=training_test_vline
res_df['tf_adj']=tf_adj
res_df['supports_lookback']=supports_lookback
res_df['supports_rank']=supports_rank

res_df_s=res_df.loc[(res_df.test_tp_acc>0.75) & (res_df.train_tp_acc>0.75) , ]

res_df_s['profit_products'] = res_df.test_profits * res_df.train_profits

res_df_s = res_df_s.sort_values( 'profit_products' ,ascending=False)

topfeats= res_df_s.loc[:,'feats'].iloc[0:100,]

topf=[]
for k in topfeats:
    f=k.split(", ")
    topf=[*topf, *f]

topf=pd.Series(topf).value_counts()
#####
features=list(topf.index[0:50].str.replace('"', '').str.replace(' ', '').str.strip() )
nfeats=6
#####

#########################################


###########
openpre="bundle_BTCUSDT_15m_1000_3_1.024_0.96_short.pkl"
with open(openpre, 'rb') as file:
    loaded_bundle = pickle.load(file)

# loaded_bundle['trades_params']['tp'] = 1.024
# loaded_bundle['trades_params']['sl'] = 0.95
# with open(openpre, 'wb') as file:
#     pickle.dump(loaded_bundle, file)


# Access the loaded model from the dictionary
trades = loaded_bundle['full_data']
y = loaded_bundle['class']
paramsused = loaded_bundle['parameters_used']
all_supports = loaded_bundle['supports']
trades_params = loaded_bundle['trades_params']
window= trades_params['window']
top_n= trades_params['top_n']
tradetype= trades_params['tradetype']
tp, sl = trades_params['tp'],trades_params['sl']
supports_lookback, supports_rank = trades_params['window'] , trades_params['top_n'] 

btcdataset=get_market_data(trades_params['symbol'] ,trades_params['tf'],api_key,api_secret)

all_supports=calc_supports_3(btcdataset,window,top_n,True)            

rf_df = loaded_bundle['useful_data']
rf_df = rf_df.loc[ (pd.Series(y).isin(['tp','sl'])).values , ]
trades=trades.loc[ (pd.Series(y).isin(['tp','sl'])).values , ]
y=y[ (pd.Series(y).isin(['tp','sl'])).values , ]
y = np.where(y == 'tp', 1, 0)

sensibility=0.45
md=3
nestimators=10
subsample=1
lr=0.05
featspertree = 1
ttratio= 0.3
###########

# prod_features:
prod_features=[	"ema_250_rate_2000", "ema_875_rate_1500", "wma_125_ratio_to_order", "ema_250_rate_250", "minmax_hl_187_rate_2000", "minmax_hl_75_rate_2000"]

training_test_vline=trades.iloc[ int( (1-ttratio) *len(trades)) ,trades.columns.get_loc("timestamp2")   ]
training_test_vline

X = rf_df.loc[:, prod_features].values

X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=ttratio, random_state=42, shuffle=False)


model = XGBClassifier(scale_pos_weight=sensibility,
                      max_depth=md,
                      n_estimators=nestimators,
                      colsample_bytree=featspertree,
                      learning_rate=lr,
                      subsample=subsample)

model.fit(X_train, y_train)
yhat = model.predict(X_test)
confusion = confusion_matrix(y_test, yhat)
confusion

X_ahead = rf_df.loc[:,prod_features].values

yhat_ahead = model.predict(X_ahead)

trades_ahead_go = trades.loc[ yhat_ahead == 1 , ]
trades_ahead_go.result.value_counts()
charter(trades_ahead_go, btcdataset, all_supports, False, training_test_vline, tradetype)


#### PREDICT

lkeys = paramsused['keys']
llabels = paramsused['labels']
lparams = paramsused['valparam']

btcdataset=get_market_data(symbol,tf,api_key,api_secret)


all_supports=calc_supports_3(btcdataset,supports_lookback,supports_rank,True)            

for i in ['deltaprice','deltatime','score','from_ratio']:
    lkeys.append(i)
    llabels.append(i)
    lparams.append(True)

prod_params={}
for i in prod_features:
    ix= llabels.index(i)
    k = lkeys[ix]
    l = lparams[ix]
    if k not in prod_params:
        prod_params[k] = [l]
    else:
        prod_params[k] = prod_params[k] + [l]

rf_df_go,y_go,trades_go = process_strategy(all_supports, btcdataset, tp, sl,tradetype, 9000, True, prod_params)

X_ahead = rf_df_go.loc[:, prod_features].values


yhat_ahead = model.predict(X_ahead)
yhat_ahead_prob = model.predict_proba(X_ahead)[:, 1]

trades_ahead_go = trades_go.loc[ yhat_ahead == 1 , ]
trades_ahead_go.result.value_counts()

charter(trades_ahead_go, btcdataset,all_supports,False,False,tradetype)


########################################

signature = '' 
for t in prod_features:
    signature += t[0]

modelname = 'model_'+ tradetype +'_'+ symbol+'_'+tf+'_'+str(supports_lookback)  +'_'+ str(supports_rank)+'_'+ str(tp)+'_'+ str(sl)+'_'+ signature


modeldict={
 'modelname':modelname,
 'model':model,
 'xgboost_params':store_xgboost_params,
 'features': prod_features,
 'paramsused':paramsused,
 'trades_params': {'tp':tp,
                 'sl':sl,
                 'symbol': symbol,
                 'tf':tf,
                 'window': supports_lookback,
                 'top_n':supports_rank ,
                 'tradetype': tradetype}
}

with open(modelname+'.pkl', 'wb') as file:
    pickle.dump(modeldict, file)


########################################


import pickle
with open('bundle_BTCUSDT_1h_401_5_1.034_0.955_wqeqwq.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

all_supports=loaded_data['supports']


with open('model_long_BTCUSDT_1h_401_5_1.034_0.955_wqeqwq.pkl', 'rb') as file:
    loaded_model_dict = pickle.load(file)

# Access the loaded model from the dictionary
model = loaded_model_dict['model']
paramsused = loaded_model_dict['paramsused']
features = loaded_model_dict['features']
xgboost_params = loaded_model_dict['xgboost_params']
trades_params = loaded_model_dict['trades_params']


symbol=trades_params['symbol']
tf=trades_params['tf']
tp = trades_params['tp']
sl = trades_params['sl']
tradetype=trades_params['tradetype']

ttratio=xgboost_params['test_size']
nfeats=xgboost_params['nfeats']
md=xgboost_params['max_depth']
nestimators= xgboost_params['n_estimators']
featspertree=xgboost_params['colsample_bytree']
lr=xgboost_params['learning_rate']
subsample =xgboost_params['subsample']

sensibility=xgboost_params['scale_pos_weight']

#### PREDICT
lkeys = paramsused['keys']
llabels = paramsused['labels']
lparams = paramsused['valparam']

for i in ['deltaprice','deltatime','score','from_ratio']:
    lkeys.append(i)
    llabels.append(i)
    lparams.append(True)

prod_params={}
for i in features:
    ix= llabels.index(i)
    k = lkeys[ix]
    l = lparams[ix]
    if k not in prod_params:
        prod_params[k] = [l]
    else:
        prod_params[k] = prod_params[k] + [l]

rf_df_go,y_go,trades_go = process_strategy(all_supports, btcdataset, tp, sl,tradetype, 6000, True, prod_params)

X_ahead = rf_df_go.loc[:, features].values


yhat_ahead = model.predict(X_ahead)

trades_ahead_go = trades_go.loc[ yhat_ahead == 1 , ]
trades_ahead_go.result.value_counts()

charter(trades_ahead_go, btcdataset,all_supports,False,False,tradetype)















from joblib import dump, load


dump(model, 'longs15m.joblib')




import pandas as pd
import numpy as np
from aux_functions import *


from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from decimal import Decimal




api_key = ''    #
api_secret = '' #
client = Client(api_key, api_secret)



ts_cutoff = 1701190036678+1

active_orders=client.futures_get_all_orders() #startTime=1643989224456, endTime=   1644548400000)
    
active_orders = pd.DataFrame(active_orders)
active_orders = active_orders.loc[ active_orders.time > ts_cutoff , : ] 
active_orders.sort_values( by=['updateTime'], inplace=True )
active_orders['price'] = pd.to_numeric(active_orders['price'])
active_orders['stopPrice'] = pd.to_numeric(active_orders['stopPrice'])
active_orders['avgPrice'] = pd.to_numeric(active_orders['avgPrice'])

active_orders = active_orders.loc[ active_orders.status == 'NEW' , :]

active_orders = active_orders.loc[ active_orders.status == 'FILLED' , :]

active_orders = active_orders.loc[ active_orders.type != 'MARKET' , :]

active_orders = active_orders.loc[ active_orders.type != 'CANCELED' , :]























