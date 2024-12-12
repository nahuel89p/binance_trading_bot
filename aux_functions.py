# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 13:17:54 2021

@author: User
"""

import os

# import calendar
import pandas as pd
# import math
import numpy as np
import datetime
import pickle

import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from binance.enums import *
import copy as copy

import datetime

from plotly.offline import plot
from plotly.subplots import make_subplots
import ccxt


def qvwma(X, weights, threshold,exp):
    retval = np.repeat(float("Nan"), weights.shape[0])
    tail = weights.shape[0]
    head = tail -1
    weightsum = 0

    while head >= 0:
        weightsum += weights[head]

        while weightsum >= threshold:
            tail -= 1
            retval[tail] = np.average(X[head:tail+1], weights= pow(weights[head:tail+1],exp) )
            weightsum -= weights[tail]
        
        head -= 1
    
    return retval


vqvwma = np.vectorize(qvwma, cache=False, signature='(n),(n),(),()->(n)')


def volume_weighted_moving_average(close_prices, volumes, window, exp):
    # Calculate the product of close prices and volumes
    volumes = volumes**exp
    
    weighted_prices = close_prices * volumes
    
    # Calculate the rolling sum of weighted prices over the specified window
    rolling_weighted_sum = weighted_prices.rolling(window=window).sum()
    
    # Calculate the rolling sum of volumes over the specified window
    rolling_volume_sum = volumes.rolling(window=window).sum()
    
    # Calculate the volume-weighted moving average (VWMA)
    vwma = rolling_weighted_sum / rolling_volume_sum
    
    return vwma


# market="BTCUSDT"
# tf

def get_market_data(market,tf,k,s):
    Client = ccxt.binance({
    'apiKey': k,
    'secret': s,
})
    Client.options = {'defaultType': 'future','adjustForTimeDifference':False,}

    try:
        btcdataset = pd.read_pickle(market+'_'+tf+'.pkl')
        tsiloc = btcdataset.columns.get_loc('timestamp')
        
        
        btcdataset=btcdataset.iloc[:-7,]
        since=btcdataset.iat[ -1 ,tsiloc]
    except:
        cdate='2020-07-07T00:00:00Z'
        since = Client.parse8601(cdate)
        btcdataset=pd.DataFrame(columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume' ]) 


    mdata=pd.DataFrame(columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume' ]) 
    while True:
        print(pd.to_datetime(since,unit='ms') )   
        
        # params = {'market_name': market,'defaultType': 'future'} 
        limit = None
        
        aver=Client.fetchOHLCV(market, timeframe=tf,since=int(since), limit=limit#, params=params
                                 )
        
        aver=pd.DataFrame(aver,columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume' ])
        present=aver.loc[:,'timestamp'].max()
        
        if present == since:
            break
                
        
        mdata=pd.concat([mdata,aver], ignore_index=True)
        
        since=mdata.loc[:,'timestamp'].max()


    mdata=mdata.drop_duplicates('timestamp')    
    mdata['time']=pd.to_datetime(mdata['timestamp'],unit='ms')
    mdata['timestamp'] = pd.to_numeric(mdata['timestamp'])

    mdata.reset_index(inplace=True)
    
    btcdataset = pd.concat([btcdataset,mdata] , ignore_index=True )
    btcdataset=btcdataset.drop_duplicates('timestamp')    
    btcdataset=btcdataset.reset_index(drop=True)    
    
    btcdataset['market'] = market
    btcdataset['timeframe'] = tf
    btcdataset['timestamp'] = pd.to_numeric(btcdataset['timestamp'])

    btcdataset.to_pickle(market+'_'+tf+'.pkl')

    return btcdataset
    
    

# btc_d= copy.deepcopy(btcdataset)
# winners=calc_supports_3(btcdataset,supports_lookback,supports_rank,True) 
# t2=supports_lookback
# rank=supports_rank

def calc_supports_3(btc_d,t2,rank,save):
    
    btc_df=copy.deepcopy(btc_d)
    
    tf = btc_df['timeframe'].values[0]
    market = btc_df['market'].values[0]
    
    btc_df = btc_df.drop(columns=['timeframe','market'])
    
    fname = 'supports_'+str(market)+'_' + tf + '_rank_'+str(rank)+ '_window_'+str(t2)+ '.pkl'
    
    try:
        winners = pd.read_pickle(fname)
        init= btc_df.loc[ btc_df.timestamp == winners.expire.max(),  ].iloc[0].name
        winners.loc[ winners.expire == winners.expire.max(),'expire'] = np.nan
    except:
        winners = pd.DataFrame(columns=['volume','close','timestamp'])
        init=t2
    
    tsiloc = btc_df.columns.get_loc('timestamp')
    # closeiloc = btc_df.columns.get_loc('close')
    initday=btc_df.iat[0,tsiloc]
    lastday=btc_df.iat[-1,tsiloc]
    length=lastday-initday

    for i in range(init,len(btc_df)):
        today=btc_df.iat[int(i),tsiloc]
        print((today-initday) / length)
        
        vols=btc_df.iloc[ i-t2:i+1,:][['volume','close','low','timestamp']]
        
        maxs=np.argsort(vols['volume'].values*-1)
        maxs=maxs[0:rank]
        
        winners0=vols.iloc[maxs]
        winners0['official'] = today
        winners = pd.concat([winners,winners0]  )
        winners.sort_values(['timestamp', 'official'] , inplace=True)
        
        winners.drop_duplicates( subset=['close','timestamp'],  inplace=True, keep='first')

    winners['timestamp2']=pd.to_datetime(winners.loc[:,'timestamp'],unit='ms')
    winners['expire']  =   btc_df.iat[-1,tsiloc]    

    winners['expire2']=pd.to_datetime(winners.loc[:,'expire'],unit='ms')
    winners['timestamp'] = pd.to_numeric(winners['timestamp'])    
    winners['official'] = pd.to_numeric(winners['official'])
    winners['valid_since'] = pd.to_datetime(winners.loc[:,'official'],unit='ms')

    winners['params'] = market + '_'+ tf + '_'+ str(t2) + '_' +  str(rank)
    
    if save==True:
        winners.to_pickle(fname)
    else:
        pass
    return winners


        
def process_strategy(winners0, btcd, tp_o,sl_o,tradetype,lookback,lookahead, params):

    finalvars = set()
    
    dosls=False
    dotps=False
    do_delta_to_last_trade = False
    
    do_vol_above_order = False
    do_vol_above_price = False
    do_vol_below_order = False
    do_vol_below_price = False
    do_volbuffers = False
    
    keys = []
    valparam = []
    labels = []
        
    supports= copy.deepcopy(winners0)
    try:
        paramstr = supports['params'].values[0]
    except:
        paramstr = supports['params']
        
    name = 'bundle' + '_' +paramstr + '_' + str(tp_o)  + '_' + str(sl_o)  + '_' + str(tradetype)

    if os.path.exists( name + '.pkl'):
        name = name + '_' + str(int(time.time()))
    else:
        pass

    if isinstance(supports, pd.DataFrame):  
        supports = supports.drop(columns=['params'])
        wclosiloc=supports.columns.get_loc('close')
        wtsiloc=supports.columns.get_loc('official')
        wexpiloc=supports.columns.get_loc('expire')
        wvoliloc=supports.columns.get_loc('volume')
    else:
        pass
    
    btc=copy.deepcopy(btcd)
    btc = btc.drop(columns=['timeframe', 'market'])

    openiloc = btc.columns.get_loc('open')
    lowiloc = btc.columns.get_loc('low')
    highiloc = btc.columns.get_loc('high')
    closeiloc = btc.columns.get_loc('close')
    
    btc['from'] = btc['close'].shift(1) 
    
    if tradetype == "short":
        tp, sl = 1-(tp_o-1), 1+(1-sl_o)
    else:
        tp, sl = tp_o, sl_o
    
    if paramstr.split('_')[1].lower() == '1h':
        safe_offset=3600000
    elif paramstr.split('_')[1].lower() == '15m':
        safe_offset = 3600000/4
    

    if lookahead==True:
        btc = btc.iloc[ -lookback: , ]
        btc.reset_index(inplace=True, drop=True)
        if tradetype =='long' and isinstance(supports, pd.DataFrame) :
            supports = supports.loc[ supports.close < btc['close'].max() , ] 
        elif tradetype =='short' and isinstance(supports, pd.DataFrame) :
            supports = supports.loc[ supports.close > btc['close'].min() , ] 
    else:
        pass
    
    sups = []
    concat_cols = {}
    for i in params:
        iparams=params[i]
        if i == 'ema' and len(iparams) > 0:
            for z in iparams:
                w=z['window']
                r=z['rate']
                rto=z['ratio_to_order']
                rtp=z['ratio_to_price']
                label = 'ema_'+str(w)
                
                if label not in concat_cols:
                    concat_cols[label] = btc['close'].shift(1).ewm(span=w, adjust=False).mean()
                else:
                    pass
                
                if rtp:
                    labelrtp = label + '_ratio_to_price'
                    concat_cols[labelrtp] = concat_cols[label]/ btc['close'].shift(1)
                    finalvars.add(labelrtp)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelrtp)  
                else:
                    pass
                
                if r > 0:
                    labelr = label + '_rate_' + str(r)
                    concat_cols[labelr] = concat_cols[label]/ concat_cols[label].shift(r)
                    finalvars.add( labelr )
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelr)  
                else:
                    pass
                
                if rto:
                    sups.append(label)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(label+'_ratio_to_order')  
                else:
                    pass
            
        elif i == 'qvwma' and len(iparams) > 0:
            for z in iparams:
                w=z['window']
                r=z['rate']
                rto=z['ratio_to_order']
                rtp=z['ratio_to_price']
                exp=z['exp']
            
                label = 'qvwma_'+str(w)+'_exp'+str(exp)
                if label not in concat_cols:
                    concat_cols[label] = qvwma( btc['close'].shift(1) , btc['volume'].shift(1), w, exp)
                else:
                    pass
                
                if rtp:
                    labelrtp = label + '_ratio_to_price'
                    concat_cols[labelrtp] = concat_cols[label] / btc['close'].shift(1)
                    finalvars.add(labelrtp)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelrtp)  
                else:
                    pass
                
                if r > 0:
                    labelr = label + '_rate_' + str(r)
                    concat_cols[labelr] = concat_cols[label] /  pd.Series(concat_cols[label]).shift(r)
                    finalvars.add( labelr )
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelr)  
                else:
                    pass
                
                if rto:
                    sups.append(label)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(label+'_ratio_to_order')  
                else:
                    pass

        elif i == 'svwma' and len(iparams) > 0:
            for z in iparams:
                w=z['window']
                r=z['rate']
                rto=z['ratio_to_order']
                rtp=z['ratio_to_price']
                exp=z['exp']
            
                label = 'svwma_'+str(w)+'_exp'+str(exp)
                if label not in concat_cols:
                    concat_cols[label] = volume_weighted_moving_average(btc['close'].shift(1) , btc['volume'].shift(1) , w, exp)
                else:
                    pass
                
                if rtp:
                    labelrtp = label + '_ratio_to_price'
                    concat_cols[labelrtp] = concat_cols[label] / btc['close'].shift(1)
                    finalvars.add(labelrtp)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelrtp)  
                else:
                    pass
                
                if r > 0:
                    labelr = label + '_rate_' + str(r)
                    concat_cols[labelr] = concat_cols[label] / pd.Series(concat_cols[label]).shift(r)
                    finalvars.add( labelr )
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelr)  
                else:
                    pass
                
                if rto:
                    sups.append(label)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(label+'_ratio_to_order')  
                else:
                    pass

                
        elif i == 'wma' and len(iparams) > 0:
            for z in iparams:
                w=z['window']
                r=z['rate']
                rto=z['ratio_to_order']
                rtp=z['ratio_to_price']
                label = 'wma_'+str(w)
                
                if label not in concat_cols:
                    weights = np.arange(1,w+1) 
                    concat_cols[label] = btc['close'].shift(1).rolling(w).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
                else:
                    pass
                
                if rtp:
                    labelrtp = label + '_ratio_to_price'
                    concat_cols[labelrtp] = concat_cols[label] / btc['close'].shift(1)
                    finalvars.add(labelrtp)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelrtp)  
                else:
                    pass
                
                if r > 0:
                    labelr = label + '_rate_' + str(r)
                    concat_cols[labelr] = concat_cols[label] /  pd.Series(concat_cols[label]).shift(r)
                    finalvars.add( labelr )
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelr)  
                else:
                    pass
                
                if rto:
                    sups.append(label)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(label+'_ratio_to_order')  
                else:
                    pass

        elif i == 'minmax_hl' and len(iparams) > 0:
            for z in iparams:
                w=z['window']
                r=z['rate']
                rto=z['ratio_to_order']
                rtp=z['ratio_to_price']
                label = 'minmax_hl_'+str(w)
                
                if label not in concat_cols:
                    min1=btc['low'].shift(1).rolling(w).min()
                    max1=btc['high'].shift(1).rolling(w).max()
                    concat_cols[label] = (min1 +max1) / 2                    
                else:
                    pass
                
                if rtp:
                    labelrtp = label + '_ratio_to_price'
                    concat_cols[labelrtp] = concat_cols[label] / btc['close'].shift(1)
                    finalvars.add(labelrtp)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelrtp)  
                else:
                    pass
                
                if r > 0:
                    labelr = label + '_rate_' + str(r)
                    concat_cols[labelr] = concat_cols[label] /  pd.Series(concat_cols[label]).shift(r)
                    finalvars.add( labelr )
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(labelr)  
                else:
                    pass
                
                if rto:
                    sups.append(label)
                    
                    keys.append(i)
                    valparam.append(z)
                    labels.append(label+'_ratio_to_order')  
                else:
                    pass

        elif i == 'last_sls':
            slsp = params['last_sls' ]
            if len(slsp)>0:
                dosls=True
            else:
                pass
    
        elif i == 'last_tps':
            tpsp = params['last_tps' ]
            if len(tpsp)>0:
                dotps=True
            else:
                pass
            
        elif i == 'vol_buffers':
            vol_buffers = params['vol_buffers' ]
            if len(vol_buffers)>0:
                do_volbuffers=True
            else:
                pass
            
        elif i ==  'delta_to_last_trade':
            if params['delta_to_last_trade' ]:
                do_delta_to_last_trade = True
            else:
                do_delta_to_last_trade = False
    
        elif i ==  'vol_above_order':
            if len(params['vol_above_order' ])>0:
                do_vol_above_order = True
    
        elif i ==  'vol_above_price':
            if len(params['vol_above_price' ])>0:
                do_vol_above_price = True
    
        elif i ==  'vol_below_order':
            if len(params['vol_below_order' ])>0:
                do_vol_below_order = True
    
        elif i ==  'vol_below_price':
            if len(params['vol_below_price' ])>0:
                do_vol_below_price = True
    
    btc = pd.concat([btc, pd.DataFrame(concat_cols )], axis=1)
    
    if isinstance(supports, dict):
        slen = len(supports['ema'])
    else:
        slen = len(supports)
        
    dfs_list = [None] * slen

    do_order_ratio_to_rolling = []
    if 'order_ratio_to_rolling' in params:
        orr = params['order_ratio_to_rolling']
        if len(orr) > 0:
            for z in orr:
                t = z['type']
                w = z['window']
                label = 'order_ratio_to_rolling_'+t+'_'+str(w)
                if t == 'high':
                    btc[label] = btc[t].shift(1).rolling(w).max() 
                elif t == 'low':
                    btc[label] = btc[t].shift(1).rolling(w).min()
                else:
                    pass
                do_order_ratio_to_rolling.append(label)
                
                keys.append('order_ratio_to_rolling')
                valparam.append(z)
                labels.append(label)  
    else: 
        pass  

    do_price_ratio_to_rolling = []
    if 'price_ratio_to_rolling' in params:
        orr = params['price_ratio_to_rolling']
        if len(orr) > 0:
            for z in orr:
                t = z['type']
                w = z['window']
                label = 'price_ratio_to_rolling'+t+'_'+str(w)
                if t == 'high':
                    btc[label] = btc[t].shift(1).rolling(w).max() 
                elif t == 'low':
                    btc[label] = btc[t].shift(1).rolling(w).min()
                else:
                    pass
                do_price_ratio_to_rolling.append(label)
                
                keys.append('price_ratio_to_rolling')
                valparam.append(z)
                labels.append(label)  
    else: 
        pass  
    
    concat_cols = {}

    if isinstance(supports, dict):
        for i in supports:
            iparams=supports[i]
            if i == 'ema' and len(iparams) > 0:
                for z in iparams:
                    w=z['window']
                    label = 'support_'+i+'_'+str(w)
                    
                    concat_cols[label] = btc['close'].shift(1).ewm(span=w, adjust=False).mean()
        
            else:
                pass
        btc = pd.concat([btc, pd.DataFrame(concat_cols )], axis=1)

        for i, r in zip(concat_cols, range(len(concat_cols))):
            supiloc = btc.columns.get_loc(i)

            btc['flag' ]=False
            if tradetype == "long":
                if lookahead==True:
                    btc.iloc[-1 , openiloc ] = btc[i]+1
                    btc.iloc[-1, lowiloc ] = btc[i]-1
                else:
                    pass
                # btc.loc[ (btc.open > c1) & (btc.low < c1)  & (btc.timestamp-safe_offset > t1) & (btc.timestamp <= t2) ,'flag' ] = True 
                btc.loc[ (btc.open > btc[i]) & (btc.low < btc[i])  ,'flag' ] = True 
            elif tradetype == "short":
                if lookahead==True:
                    btc.iloc[-1 ,openiloc ] = btc[i]-1
                    btc.iloc[-1,highiloc ] = btc[i]+1
                else:
                    pass          
                # btc.loc[ (btc.open < c1) & (btc.high > c1) & (btc.timestamp-safe_offset > t1) & (btc.timestamp <= t2) ,'flag' ] = True 
                btc.loc[ (btc.open < btc[i]) & (btc.high > btc[i])  ,'flag' ] = True
            else:
                pass
                        
            btc['flag_tp'] = False
            btc['flag_sl'] = False
        
            btc['result'] = "no_trade"
                
            
            btc['order_price'] = btc[i]
            btc['vigente'] = 0
        
            resiloc = btc.columns.get_loc('result')
            flagiloc = btc.columns.get_loc('flag')
        
            timestampiloc = btc.columns.get_loc('timestamp')
            
        
            while btc['flag' ].sum()!=0:
                ix1 = btc[btc.flag==True].first_valid_index()
                
                if tradetype=="long":
                    btc.loc[ btc.high > btc.iloc[ix1, supiloc ]*tp, 'flag_tp'] = True
                    btc.loc[ btc.low < btc.iloc[ix1, supiloc ]*sl, 'flag_sl'] = True    
                else:
                    btc.loc[ btc.high > btc.iloc[ix1, supiloc ]*sl, 'flag_sl'] = True
                    btc.loc[ btc.low < btc.iloc[ix1, supiloc ]*tp, 'flag_tp'] = True                

        
                ixtp = btc.iloc[ ix1+1:,  ].loc[btc.flag_tp==True, ].first_valid_index()
                ixsl = btc.iloc[ ix1:,  ].loc[btc.flag_sl==True, ].first_valid_index()
                
                ixtp = np.inf if ixtp == None else ixtp
                ixsl = np.inf if ixsl == None else ixsl
                                        
                if ixtp < ixsl:
                    btc.iloc[ix1 ,resiloc ]="tp"
                    btc.iloc[ ix1:min(ixtp,ixsl)+1 ,  flagiloc]=False
                                        
                    
                elif ixtp > ixsl:
                    btc.iloc[ix1 ,resiloc ]="sl"
                    btc.iloc[ ix1:min(ixtp,ixsl)+1 ,  flagiloc]=False
                                        
                    
                elif ixtp == ixsl == np.inf:
                    btc.iloc[ix1 ,resiloc ]="open_trade"
                    btc.iloc[  ix1: ,  flagiloc]=False
                    btc.iloc[  ix1+1: ,  resiloc]="no_trade"
                    
                else:
                    btc.iloc[ix1 ,resiloc ]="sl" 
                    btc.iloc[ ix1:ix1+1 ,  flagiloc]=False
                
            #print(c1)    
            
            trades0 = btc.loc[ btc.result!='no_trade', : ]
            dfs_list[r] = trades0

    else:
        for i in range(slen):
            #concat_cols = {}
            btc['delta_to_last_trade'] = 0        
            t1=int(supports.iat[ (i), wtsiloc])
            #t2=int(supports.iat[ (i), wexpiloc])
            c1=supports.iat[ (i), wclosiloc]  
        
            if 'abs_vol' in params:
                if params['abs_vol']== True:
                    v1=supports.iat[ (i), wvoliloc]
                    btc['abs_vol'] = v1
                    finalvars.add('abs_vol')
                    
                    keys.append('abs_vol')
                    valparam.append(True)
                    labels.append('abs_vol')  
                else:
                    pass
            else: 
                pass
        
            btc['flag' ]=False
            if tradetype == "long":
                if lookahead==True:
                    btc.iloc[-1 , openiloc ] = c1+1
                    btc.iloc[-1, lowiloc ] = c1-1
                else:
                    pass
                # btc.loc[ (btc.open > c1) & (btc.low < c1)  & (btc.timestamp-safe_offset > t1) & (btc.timestamp <= t2) ,'flag' ] = True 
                btc.loc[ (btc.open > c1) & (btc.low < c1)  & (btc.timestamp > t1) ,'flag' ] = True 
            elif tradetype == "short":
                if lookahead==True:
                    btc.iloc[-1 ,openiloc ] = c1-1
                    btc.iloc[-1,highiloc ] = c1+1
                else:
                    pass          
                # btc.loc[ (btc.open < c1) & (btc.high > c1) & (btc.timestamp-safe_offset > t1) & (btc.timestamp <= t2) ,'flag' ] = True 
                btc.loc[ (btc.open < c1) & (btc.high > c1) & (btc.timestamp > t1)  ,'flag' ] = True
            else:
                pass
                
            btc['flag_tp'] = False
            btc['flag_sl'] = False
        
            btc['result'] = "no_trade"
                
            if tradetype=="long":
                btc.loc[ btc.high > c1*tp, 'flag_tp'] = True
                btc.loc[  btc.low < c1*sl, 'flag_sl'] = True    
            else:
                btc.loc[ btc.high > c1*sl, 'flag_sl'] = True
                btc.loc[  btc.low < c1*tp, 'flag_tp'] = True                
            
            btc['order_price'] = c1
            btc['vigente'] = t1
        
            resiloc = btc.columns.get_loc('result')
            flagiloc = btc.columns.get_loc('flag')
        
            lastrtradeiloc = btc.columns.get_loc('delta_to_last_trade')
            timestampiloc = btc.columns.get_loc('timestamp')
        
            last_trade= copy.deepcopy(t1)           
            
            while btc['flag' ].sum()!=0:
                ix1 = btc[btc.flag==True].first_valid_index()
        
                ixtp = btc.iloc[ ix1+1:,  ].loc[btc.flag_tp==True, ].first_valid_index()
                ixsl = btc.iloc[ ix1:,  ].loc[btc.flag_sl==True, ].first_valid_index()
                
                ixtp = np.inf if ixtp == None else ixtp
                ixsl = np.inf if ixsl == None else ixsl
                
                if do_volbuffers:                
                    for z in vol_buffers:
                        label='vol_buffer_' + str(z)
                        btc[label ] = 0
                        finalvars.add(label)
                        volbufferiloc = btc.columns.get_loc(label)
                        if tradetype == 'long':
                            btc.iloc[ ix1 ,  volbufferiloc] = btc.iloc[ ix1-z:ix1+1, : ].loc[ (btc.close > c1) & (btc.close < c1 * tp) , 'volume'  ].sum()
                        else:
                            btc.iloc[ ix1 ,  volbufferiloc] = btc.iloc[ ix1-z:ix1+1, : ].loc[ (btc.close < c1) & (btc.close > c1 * tp) , 'volume'  ].sum()
                            
                        keys.append('vol_buffers')
                        valparam.append(z)
                        labels.append(label)  
                else:
                    pass
                        
                if ixtp < ixsl:
                    btc.iloc[ix1 ,resiloc ]="tp"
                    btc.iloc[ ix1:min(ixtp,ixsl)+1 ,  flagiloc]=False
                    
                    if dotps:
                        for z in tpsp:
                            label = 'last_tp_' + str(z)
                            btc[label ] = 0
                            finalvars.add(label)
                            lasttpsiloc = btc.columns.get_loc(label)
                            btc.iloc[ ix1 ,  lasttpsiloc] = btc.iloc[ix1-z:ix1 ,  ].loc[ btc.result=="tp", ].count()[0]
                            
                            keys.append('last_tps')
                            valparam.append(z)
                            labels.append(label)  
                    else:
                        pass
                        
                    if dosls:
                        for z in slsp:
                            label = 'last_sl_' + str(z)
                            btc[label] = 0
                            finalvars.add(label)
                            lastslsliloc = btc.columns.get_loc(label)
                            btc.iloc[ ix1 ,  lastslsliloc] = btc.iloc[ix1-z:ix1 ,  ].loc[ btc.result=="sl", ].count()[0]
                            
                            keys.append('last_sls')
                            valparam.append(z)
                            labels.append(label)  
                    else:
                        pass
                    
                    if do_delta_to_last_trade:
                        btc.iloc[ ix1 ,  lastrtradeiloc] =  btc.iat[ ix1 ,  timestampiloc] - last_trade
                        last_trade = btc.iat[ ix1 ,  timestampiloc]
                    else:
                        pass
                    
                elif ixtp > ixsl:
                    btc.iloc[ix1 ,resiloc ]="sl"
                    btc.iloc[ ix1:min(ixtp,ixsl)+1 ,  flagiloc]=False
        
                    if dotps:
                        for z in tpsp:
                            label = 'last_tp_' + str(z)
                            btc[label ] = 0
                            finalvars.add(label)
                            lasttpsiloc = btc.columns.get_loc(label)
                            btc.iloc[ ix1 ,  lasttpsiloc] = btc.iloc[ix1-z:ix1 ,  ].loc[ btc.result=="tp", ].count()[0]
                            
                            keys.append('last_tps')
                            valparam.append(z)
                            labels.append(label)  
                    else:
                        pass
                        
                    if dosls:
                        for z in slsp:
                            label = 'last_sl_' + str(z)
                            btc[label] = 0
                            finalvars.add(label)
                            lastslsliloc = btc.columns.get_loc(label)
                            btc.iloc[ ix1 ,  lastslsliloc] = btc.iloc[ix1-z:ix1 ,  ].loc[ btc.result=="sl", ].count()[0]
                            
                            keys.append('last_sls')
                            valparam.append(z)
                            labels.append(label)  
                    else:
                        pass
                                
                    if do_delta_to_last_trade:
                        btc.iloc[ ix1 ,  lastrtradeiloc] =  btc.iat[ ix1 ,  timestampiloc] - last_trade
                        last_trade = btc.iat[ ix1 ,  timestampiloc]
                    else:
                        pass
                    
                elif ixtp == ixsl == np.inf:
                    btc.iloc[ix1 ,resiloc ]="open_trade"
                    btc.iloc[  ix1: ,  flagiloc]=False
                    btc.iloc[  ix1+1: ,  resiloc]="no_trade"
                    if do_delta_to_last_trade:
                        btc.iloc[ ix1 ,  lastrtradeiloc] =  btc.iat[ ix1 ,  timestampiloc] - last_trade
                        last_trade = btc.iat[ ix1 ,  timestampiloc]                    
                    else:
                        pass
                    
                else:
                    btc.iloc[ix1 ,resiloc ]="sl" 
                    btc.iloc[ ix1:ix1+1 ,  flagiloc]=False
                    if do_delta_to_last_trade:
                        btc.iloc[ ix1 ,  lastrtradeiloc] =  btc.iat[ ix1 ,  timestampiloc] - last_trade
                        last_trade = btc.iat[ ix1 ,  timestampiloc]                    
                    else:
                        pass
            trades0 = btc.loc[ btc.result!='no_trade', : ]
            dfs_list[i] = trades0
        
    trades = pd.concat(dfs_list , ignore_index=False )
    
    if tradetype == "long":
        trades['tradetype'] = "long"
    else:
        trades['tradetype'] = "short"
        
    trades['timestamp2']=pd.to_datetime(trades['timestamp'],unit='ms')
    trades.sort_values('timestamp', inplace=True)
    
    trades.dropna(inplace=True)
            
    vigenteiloc=trades.columns.get_loc('vigente')
    
    order_price_iloc = trades.columns.get_loc('order_price')
    from_iloc = trades.columns.get_loc('from')
        
    lentrades = len(trades)
    
    labelsresdict = {}
    
    if do_vol_above_order or do_vol_above_price or do_vol_below_order or do_vol_below_price:
        t1s = [None] * lentrades
        p1_orders = [None] * lentrades
        p1_prices = [None] * lentrades
        
        for i in range(lentrades):
            t1=trades.iat[i,vigenteiloc]
            p1_order=trades.iat[i,order_price_iloc]    
            p1_price =trades.iat[i,from_iloc]
            
            t1s[i] = t1
            p1_orders[i] = p1_order
            p1_prices[i] = p1_price
    else:
        pass
            
    if do_vol_above_order:
        for p in params['vol_above_order' ]:
            label1='vol_above_order_'+str(p)
            res1=[None] * lentrades
            for i in range(lentrades):
                t1=t1s[i]
                p1_order=p1_orders[i]
                
                r1=supports.loc[(supports.close>p1_order) & (supports.close<p1_order*p) & (supports.timestamp < t1), 'volume' ].count() 
                res1[i]=r1
            labelsresdict[label1] = res1
            finalvars.add( label1)
            
            keys.append('vol_above_order')
            valparam.append(p)
            labels.append(label1)  
    else:
        pass
    
    if do_vol_above_price:
        for p in params['vol_above_price' ]:
            label2='vol_above_price_'+str(p)
            res2=[None] * lentrades
            for i in range(lentrades):
                t1=t1s[i]
                p1_price=p1_prices[i]

                r1=supports.loc[(supports.close>p1_price) & (supports.close<p1_price*p) & (supports.timestamp < t1), 'volume' ].count() 
                res2[i]=r1
                
            labelsresdict[label2] = res2
            finalvars.add( label2)
            
            keys.append('vol_above_price')
            valparam.append(p)
            labels.append(label2)  
    else:
        pass
            
    if do_vol_below_price:
        for p in params['vol_below_price' ]:
            label3='vol_below_price_'+str(p)
            res3=[None] * lentrades
            for i in range(lentrades):
                t1=t1s[i]
                p1_price=p1_prices[i]

                r1=supports.loc[(supports.close<p1_price) & (supports.close>p1_price*p) & (supports.timestamp < t1), 'volume' ].count()
                res3[i]=r1
                
            labelsresdict[label3] = res3
            finalvars.add( label3)
            
            keys.append('vol_below_price')
            valparam.append(p)
            labels.append(label3)  
            
    else:
        pass
    
    if do_vol_below_order:
        for p in params['vol_below_order' ]:
            label4='vol_below_order_'+str(p)
            res4=[None] * lentrades
            for i in range(lentrades):
                t1=t1s[i]
                p1_order=p1_orders[i]

                r1=supports.loc[(supports.close<p1_order) & (supports.close>p1_order*p) & (supports.timestamp < t1), 'volume' ].count() 
                res4[i]=r1
                
            labelsresdict[label4] = res4
            finalvars.add( label4)
            
            keys.append('vol_below_order')
            valparam.append(p)
            labels.append(label4)  
    else:
        pass
    
    trades = trades.assign(**labelsresdict)

    trades['deltaprice'] = (trades['from']-trades['order_price'])/trades['from']
    finalvars.add( 'deltaprice')
    keys.append('deltaprice')
    valparam.append(True)
    labels.append('deltaprice') 
    
    trades['deltatime'] =   (trades['timestamp']-trades['vigente'])
    finalvars.add( 'deltatime')
    keys.append('deltatime')
    valparam.append(True)
    labels.append('deltatime')  
    
    trades['score'] = 100000000 * (1*(trades['volume'] **1)  / trades['deltatime']) *   (abs(trades['deltaprice'])  **2)  #/ ((winners['close'] - winners.iat[0 , 1]) / np.arange(0, len(winners), 1, dtype=int))**2   
    finalvars.add( 'score')
    keys.append('score')
    valparam.append(True)
    labels.append('score')  
    
    for i in sups:
        trades[ i+'_ratio_to_order'] = trades[i] /trades['order_price']
        finalvars.add(i+'_ratio_to_order')    

    for i in do_order_ratio_to_rolling:
        trades[i] = trades[i]/trades['order_price']
        finalvars.add(i)
        
    for i in do_price_ratio_to_rolling:
        trades[i] = trades[i]/trades['from']
        finalvars.add(i)     
        
    clase=trades.loc[ : ,'result'].values

    finalvars=list(finalvars)
    output=trades.loc[ : ,  finalvars ]
    
    output = output.apply(pd.to_numeric, axis=1)
        
    paramsused = {   
        'keys': keys, 
        'valparam': valparam, 
        'labels': labels 
        }
    print("enchulado")
    if lookahead:
        return output,clase, trades
    else: 

        dumpit={
            "trades":output,
            'supports':winners0,
             "class": clase,
             "useful_data":output,
             "full_data":trades,
             "parameters":params,
             "parameters_used":paramsused,
             'name':name,
             'trades_params':{'tp':tp_o,
                              'sl':sl_o,
                              'symbol':paramstr.split('_')[0],
                              'tf':paramstr.split('_')[1],
                              'window':int(paramstr.split('_')[2]),
                              'top_n':int(paramstr.split('_')[3]),
                              'tradetype': tradetype}
                     }

        with open(name+'.pkl', 'wb') as file:
            pickle.dump(dumpit, file)
        
        return output,clase, trades, paramsused
        












from collections import Counter

def charter(trades_df_p, btcdataset_p, all_supports_p, plot_lines, training_test_vline, tradetype):
    
    all_supports = copy.deepcopy(all_supports_p)
    trades_df =   copy.deepcopy(trades_df_p)
    btcdataset = copy.deepcopy(btcdataset_p)
    
    mints=trades_df.timestamp.min()
    
    if tradetype=="long":
        trades_df = trades_df.loc[np.invert((trades_df.order_price> btcdataset['close'].tail(1).values[0] )&(trades_df.timestamp== trades_df.timestamp.max())), ]
        
        btc_go = btcdataset.loc[ btcdataset.timestamp>mints  ,: ]
        
        trades_df.loc[(trades_df.result== "open_trade" )&(trades_df.timestamp== btcdataset.timestamp.max())&(trades_df.order_price> btcdataset['open'].tail(1).values[0] ),'result' ]="supress"
    
        trades_df.loc[(trades_df.result== "open_trade" )&(trades_df.timestamp== btcdataset.timestamp.max())&(trades_df.order_price< btcdataset['low'].tail(1).values[0] ),'result' ]="resting_order"
    elif tradetype=="short":
        trades_df = trades_df.loc[np.invert((trades_df.order_price< btcdataset['close'].tail(1).values[0] )&(trades_df.timestamp== trades_df.timestamp.max())), ]
        
        btc_go = btcdataset.loc[ btcdataset.timestamp>mints  ,: ]
        
        trades_df.loc[(trades_df.result== "open_trade" )&(trades_df.timestamp== btcdataset.timestamp.max())&(trades_df.order_price< btcdataset['open'].tail(1).values[0] ),'result' ]="supress"
    
        trades_df.loc[(trades_df.result== "open_trade" )&(trades_df.timestamp== btcdataset.timestamp.max())&(trades_df.order_price> btcdataset['high'].tail(1).values[0] ),'result' ]="resting_order"
        
        
    all_supports['timestamp'] = all_supports['timestamp'].clip(mints,9999999999999999999)
    all_supports['timestamp2']=pd.to_datetime(all_supports['timestamp'],unit='ms')
    all_supports['expire2']=pd.to_datetime(all_supports['expire'],unit='ms')
    
    startiloc = all_supports.columns.get_loc('timestamp2')
    endiloc = all_supports.columns.get_loc('expire2')
    closeiloc = all_supports.columns.get_loc('close')
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if plot_lines==True:
        for i in range(len(all_supports)):
            print(i)
            fig.add_shape(type='line',
                            x0=all_supports.iat[i,startiloc] ,
                            y0=all_supports.iat[i,closeiloc],
                            x1=all_supports.iat[i,endiloc],
                            y1=all_supports.iat[i,closeiloc],
                            line=dict(color='Black',),
                            xref='x',
                            yref='y'
                            
        )
    else:
        pass
    
    print("kek")
    if training_test_vline != False:
        fig.add_vline(x=training_test_vline, line_width=3, line_dash="dash", line_color="red")
    else:
        pass
    fig.add_trace(go.Candlestick(x=btc_go['time'],
                    open=btc_go['open'],
                    high=btc_go['high'],
                    low=btc_go['low'],
                    close=btc_go['close'],
                    increasing_line_color= 'blue', decreasing_line_color= '#696969', name = 'BTCUSD 60m'))

    
    # for i in range(len(trades)):
    
    fig.add_trace(go.Scatter(x=trades_df.loc[trades_df.result=="tp" ,'timestamp2'], y=trades_df.loc[ trades_df.result=="tp",'order_price'],
                        mode='markers',
                        name='profit',
                        marker=dict(
                color='Chartreuse',
                size=7,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    
    fig.add_trace(go.Scatter(x=trades_df.loc[trades_df.result=="sl" ,'timestamp2'], y=trades_df.loc[ trades_df.result=="sl",'order_price'],
                        mode='markers',
                        name='loss',
                        marker=dict(
                color='Red',
                size=7,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    fig.add_trace(go.Scatter(x=trades_df.loc[trades_df.result=="open_trade" ,'timestamp2'], y=trades_df.loc[ trades_df.result=="open_trade",'order_price'],
                        mode='markers',
                        name='open_trade',
                        marker=dict(
                color='Yellow',
                size=9,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    fig.add_trace(go.Scatter(x=trades_df.loc[trades_df.result=="indetermined" ,'timestamp2'], y=trades_df.loc[ trades_df.result=="indetermined",'order_price'],
                        mode='markers',
                        name='indetermined',
                        marker=dict(
                color='Black',
                size=9,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    fig.add_trace(go.Scatter(x=trades_df.loc[trades_df.result=="resting_order" ,'timestamp2'], y=trades_df.loc[ trades_df.result=="resting_order",'order_price'],
                        mode='markers',
                        name='resting_order',
                        marker=dict(
                color='Blue',
                size=9,
                line=dict(
                    color='Black',
                    width=1
                ))))    
        
    fig.update_layout(xaxis_rangeslider_visible=False, )
    fig.update_yaxes(type='log') # linear range
    
    fig.update_layout(
        font_color = 'white',
        autosize=False,
        width=1900,
        height=900,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4
        ),
        paper_bgcolor="black",
        plot_bgcolor ='#1f1f1f'
    )
    fig.update_xaxes(
    linecolor='lightgrey',
    gridcolor='lightgrey'
    )
    fig.update_yaxes(
        linecolor='lightgrey',
        gridcolor='lightgrey'
    )

    fig.show()
    
    plot(fig)
               

    
    
API_KEY = ''    #
API_SECRET = '' #

    

    
    