from iFinDPy import *
import datetime as dt
import time
import datetime
import pandas as pd
import numpy as np
import talib

def initialize(account):
    account.periods=10                          # 持有日期上限
    account.hold={}                                # 记录持有天数情况

    account.holdSl=[]                             # 记录短期波段的持有情况
    account.holdHb=[]                           # 记录超短追涨的持有情况

    account.security = '000016.SH'        # 大盘风控使用上证指数
    account.maxStock=20                      # 每日最多选择出来的股票数量
    account.defend=5                            # 大盘风控值

    account.execPoint=['1000','1030','1100','1129','1400','1430','1456']    #卖点判断时刻
    # 短期波段策略的选股条件
    condition_sl='''
    5日的均线>10日的均线,
    a股市值(不含限售股)>=30亿元且a股市值(不含限售股)<=100亿元,
    股票简称不包含st,
    均线多头排列,
    市盈率(pe)>0倍,
    非停牌
    '''
    # 超短追涨的选股条件
    condition_hb='''
    昨天的涨停,
    非新股,
    股票简称不包含st,
    非停牌
    '''
    # 获取wencai的数据，并用不同字段记录
    # sl:短线波段
    get_iwencai(condition_sl,"sl")
    # hb:超短追涨
    get_iwencai(condition_hb,"hb")
    return

# 盘前处理函数
def before_trading_start(account,data):
    # 风险判断
    #account.defend=secureRatio(account,data)
    account.defend=5
    # 数据整理
    account.hb=pd.DataFrame(
        {
            "symbol":account.hb,
            "buypoint":np.zeros(len(account.hb)),
            "shareCapital":np.zeros(len(account.hb)),
            "order_id":np.zeros(len(account.hb))
        })
    return

# 盘内每tick处理函数
def handle_data(account,data):
    # 大盘风控
    if account.defend<1:
        #平仓
        log.info("大盘风控平仓")
        pourAll(account)
    else:
        if get_time()=="0930":
            # # 9:30调用每日短线波段操作函数
            # for stock in account.sl:
            #     a_buyCheck(stock,account)
            # 9:30处理集合竞价
            # 购入高优先级股票
            for i in range(len(account.hb)):
                account.hb.ix[i,'buypoint']=b_buyCheck(account.hb.ix[i,'symbol'],data)
                account.hb.ix[i,'shareCapital']=b_getShareCapital(account.hb.ix[i,'symbol'])

            account.hb=account.hb.sort_values(by=['buypoint','shareCapital'],ascending=[False,False])
            account.hb=account.hb[0:min(6,len(account.hb))]

            for i in account.hb.index:
                if account.hb.ix[i,'buypoint']==1 and len(account.positions)<account.maxStock:
                    atr = getATR(account.hb.ix[i,'symbol'])
                    amount = int(account.cash/(atr[-1]*100))
                    stock = account.hb.ix[i,'symbol']
                    log.info("超短高优先买入"+stock+",数量"+str(amount))
                    trade_value(stock,amount)
                    account.holdHb.append(stock)

        if get_time()=="0932":
            # 卖出符合条件的A
            for stock in account.holdSl:
                if stock in account.hold:
                    a_sellCheck(stock, account)
            # 风险控制
            a_riskDefend(account)
        if get_time()=="0933":
            for stock in account.hold:
                delayCheck(stock,data,account)
        # 9:35购入低优先级股票
        if get_time()=="0935":
            for i in account.hb.index :
                if account.hb.ix[i,'buypoint']==0 and len(account.positions)<account.maxStock:
                    atr = getATR(account.hb.ix[i,'symbol'])
                    if not isNaN(atr[-1]):
                        amount = int(account.cash/(atr[-1]*100))
                        trade_value(account.hb.ix[i,'symbol'],amount)
                        log.info("超短低优先买入"+account.hb.ix[i,'symbol']+",数量"+str(amount))
                        account.holdHb.append(account.hb.ix[i,'symbol'])
                    
        # 在execPoint进行打板策略风控和售出
        if get_time() in account.execPoint:
            for stock in account.holdHb:
                b_sellCheck(stock,data,account)
    return

# 盘后处理函数
def after_trading_end(account, data):
    # 持股时间增加
    for stock in list(account.positions):
        if stock not in account.hold:
            account.hold[stock] = 0
        else:
            account.hold[stock] += 1
    return

# =======================
# 公共函数（util）
# ---------------------------------------
# 交易统一用交易函数，包含了撤单功能
def trade_target(stock,aim):
    # 交易到目标股数
    id=order_target(stock,aim)
    orders = get_open_orders(id)
    if orders and len(orders)>1:
        cancel_order(orders)
    return

def trade_value(stock,value):
    # 交易目标金额
    id=order_value(stock,value)
    orders = get_open_orders(id)
    if orders and len(orders)>1:
        cancel_order(orders)
    return

def trade_amount(stock,amount):
    # 交易目标数量
    id=order(stock,amount)
    orders = get_open_orders(id)
    if orders and len(orders)>1:
        cancel_order(orders)
    return

# 清仓单只股票
# 这里还是考虑了所有股票的
def pourStock(stock,account):
    trade_target(stock,0)
    if stock in account.hold:
        account.hold.pop(stock)

# 清仓全部股票
# 由于当日买入的不能进行出手，所以从account.hold里面选择股票
def pourAll(account):
    temp=[]
    for i in account.hold:
        stock=account.positions[i].symbol
        trade_target(stock,0)
        temp.append(stock)
    account.hold={}
    account.holdSl=list(set(account.holdSl) ^ set(temp))
    account.holdHb=list(set(account.holdHb) ^ set(temp))
    return

# 定位函数
def dictLoc(dict,value):
    i=0
    for params in dict:
        if value==params:
            return i
        i+=1
    return -1
# RSI计算
def getRSI(df):
    return 100-100/(1+rs(df))
def rs(df):
    up=df['up'][1:].mean()
    down=df['down'][1:].mean()
    if not down==0:
        return -df['up'][1:].mean()/df['down'][1:].mean()
    else:
        return 0

# ATR计算
def getATR(stock):
    price = history(stock, ['close', 'high', 'low'], 20, '1d', False, 'pre', is_panel=1)
    high = price['high']
    low = price['low']
    close = price['close']
    return talib.ATR(np.array(high), np.array(low), np.array(close), timeperiod=14)

#判断是否发生上、下穿
def checkthrough(a,b,terms=10):
    #需要判断之前是否发生上穿
    if a[len(a)-1]>b[len(b)-1]:
        for i in range(min(terms-1,len(a)-1,len(b)-1)):
            if a[len(a)-i-1]<b[len(b)-i-1]:
                return 1,i
    #需要判断之前是否发生下穿
    elif a[len(a)-1]<b[len(b)-1]:
        for i in range(min(terms-1,len(a)-1,len(b)-1)):
            if a[len(a)-i-1]>b[len(b)-i-1]:
                return -1,i
    return 0,-1

#获取stock股票过去terms期,以step为步长的的收盘价
def getHistory(stock, terms,start=0,step='1d'):
    close=history(stock, ['close'], terms+start, step, True,None)['close']
    return close[0:terms]

#持仓到期处理
def delayCheck(stock,data,account):
    if (stock in account.hold)and (account.hold[stock] > account.periods):
        log.info("持有到期卖出:"+stock)
        trade_target(stock, 0)
        
        if stock in account.holdSl:
            #account.holdSl.pop(stock)
            account.holdSl.remove(stock)
        if stock in account.holdHb:
            #account.holdHb.pop(stock)
            account.holdHb.remove(stock)

#大盘安全函数
def secureRatio(account,data):
    #获取上证50指数成分股代码
    stock_list= get_index_stocks(account.security)
    #stock_list= get_index_stocks('000016.SH')
    #如果个股CCI指标显示在0-30之间，买入！
    close = data.history(stock_list, 'close', 120, '60m', True, fq='pre' ,is_panel = 0)
    high = data.history(stock_list, 'high', 120, '60m', True, fq='pre' ,is_panel = 0)
    low = data.history(stock_list, 'low', 120, '60m', True, fq='pre' ,is_panel = 0)
    d=0
    for i in stock_list:
        dictMerged=dict(close[i])
        dictMerged.update(high[i])
        dictMerged.update(low[i])
        dictMerged['tp'] = ((dictMerged['close']+dictMerged['low']+dictMerged['high'])/3)
        df = pd.DataFrame(dictMerged['tp'])
        if len(df)>0 and len(list(df.iloc[-1]))>0:
            TP=list(df.iloc[-1])[0]
            ma60=pd.rolling_mean(close[i],60)
            MA=list(ma60.iloc[-1])[0]
            df = pd.DataFrame(close[i])
            md1=abs(ma60-df)
            md=pd.rolling_mean(md1,60)
            MD=list(md.iloc[-1])[0]
            CCI=((TP-MA)/MD)/0.015
            if CCI>100:
                d=d+1
    return d

# 时间获取用封装的函数
# 获取当前日期
def get_date():
    return get_datetime().strftime("%Y%m%d")

# 获取星期几
def get_weekday():
    date = get_date()
    return datetime.datetime.strptime(date, "%Y%m%d").weekday()+1

# 获取时间
def get_time():
    datatime=get_datetime()
    datatime=pd.to_datetime(datatime, unit='s')
    datatime=datatime.strftime('%Y-%m-%d %H:%M:%S')
    timeArray=time.strptime(datatime,"%Y-%m-%d %H:%M:%S")
    return time.strftime("%H%M", timeArray)
# =======================
# 策略A（shortLine）
# ---------------------------------------
# 买入判断
def a_buyCheck(stock, account):
    buypoint = a_condition_MA_b(stock)+a_condition_Flow_b(stock)+a_condition_Volume_b(
        stock)+a_condition_KDJ_b(stock)+a_condition_WeekTor_b(stock)

    if buypoint > 3:
        atr = getATR(stock)
        amount = int(account.cash/(atr[-1]*100))
        trade_amount(stock,amount)              #按量买入
        account.holdSl.append(stock)            #添加记录
        log.info("波段买入"+stock+"，数量", amount)
    return
# MA条件
def a_condition_MA_b(stock):
    # 符合MA5>MA10>MA20
    price = history(stock, ['open', 'close'], 20, '1d', False, 'pre', is_panel=1)
    close = price.iloc[:, 0]
    MA5 = talib.MA(np.array(close), timeperiod=5)
    MA10 = talib.MA(np.array(close), timeperiod=10)
    MA20 = talib.MA(np.array(close), timeperiod=20)

    if(MA5[-1] > MA10[-1]) and (MA10[-1] > MA20[-1]):
        return 1
    else:
        return 0
# 资金净流条件
def a_condition_Flow_b(stock):
    date = get_date()
    j = get_weekday()
    # 连续两周主力资金净流入大于0
    delta = datetime.timedelta(days=j)
    start1 = (get_last_datetime()-delta).strftime("%Y%m%d")
    # flow1为从start1到date为止的换手率
    flow1 = get_money_flow([stock], start1, date, ['net_flow_rate'], count=None, is_panel=0)
    this_week = sum(flow1[stock].net_flow_rate)

    start2 = (get_last_datetime()-datetime.timedelta(days=j+7)).strftime("%Y%m%d")
    end2 = (get_last_datetime()-datetime.timedelta(days=j+1)).strftime("%Y%m%d")
    # flow1为从start1到date为止的换手率
    flow2 = get_money_flow([stock], start2, end2, ['net_flow_rate'], count=None, is_panel=0)
    last_week = sum(flow2[stock].net_flow_rate)

    if (this_week > 0) and (last_week > 0):
        return 1
    else:
        return 0
# 成交量条件
def a_condition_Volume_b(stock):
    # 当前周成交量大于上一周成交量的2倍
    date = get_date()
    # 获取历史周级数据
    weekdata = get_candle_stick(stock, end_date=date, fre_step='week', fields=[
                                'volume'], skip_paused=False, fq='pre', bar_count=20, is_panel=1)
    volume = weekdata.iloc[:, 0]
    if volume[-1] > 2*volume[-2]:
        return 1
    else:
        return 0
# KDJ条件
def a_condition_KDJ_b(stock):
    price = history(stock, ['close', 'high', 'low'], 20, '1d', False, 'pre', is_panel=1)
    high = price['high']
    low = price['low']
    close = price['close']
    K, D = talib.STOCH(np.array(high), np.array(low),
                       np.array(close), 9, 3, 0, 3, 0)
    if(K[-1] > K[-2]) and (K[-2] < D[-2]) and (K[-1] > D[-1]):
        return 1
    else:
        return 0
# 换手条件
def a_condition_WeekTor_b(stock):
    # 周换手率超过15%，符合则赋值1，否则赋值0
    j = get_weekday()
    price = history(stock, ['turnover_rate'], 20, '1d', False, 'pre', is_panel=1)
    turnover_rate = price['turnover_rate']
    weektor = sum(turnover_rate[-j:])
    if(weektor > 0.15):
        return 1
    else:
        return 0
# ---------------------------------------
# 卖出判断
def a_sellCheck(stock, account):
    sellPoint = a_condition_MA_s(
        stock)+a_condition_Flow_s(stock)+a_condition_RSI_s(stock)
    temp=[]
    if sellPoint > 1:
        if stock in account.holdSl:
            # 平仓
            trade_target(stock,0)
            log.info("波段卖点卖出"+stock)
            temp.append(stock)
        account.holdSl=list(set(account.holdSl) ^ set(temp))
    return
# MA条件
def a_condition_MA_s(stock):
    # 符合MA5下穿MA10
    price = history(stock, ['close'],
                    20, '1d', False, 'pre', is_panel=1)
    close = price['close']

    MA5 = talib.MA(np.array(close), timeperiod=5)
    MA10 = talib.MA(np.array(close), timeperiod=10)
    if checkthrough(MA5,MA10)[0] ==-1:
        return 1
    else:
        return 0
# 资金净流条件
def a_condition_Flow_s(stock):
    # 周主力资金净流入小于0
    delta = datetime.timedelta(days=get_weekday())
    start = (get_datetime() - delta).strftime("%Y%m%d")
    flow = get_money_flow([stock], start, get_date(), [
        'net_flow_rate'], count=None, is_panel=0)
    this_week = sum(flow[stock].net_flow_rate)
    if this_week < 0:
        return 1
    else:
        return 0
# RSI条件
def a_condition_RSI_s(stock):
    # 日线RSI出现卖点信号
    price = history(stock, ['close'],
                    20, '1d', False, 'pre', is_panel=1)
    close = price['close']
    RSI1 = talib.RSI(np.array(close), timeperiod=5)
    RSI2 = talib.RSI(np.array(close), timeperiod=13)
    if(RSI1[-1] < RSI1[-2]) and (RSI1[-1] < RSI2[-1]) and (RSI1[-2] > RSI2[-2]):
        return 1
    else:
        return 0
# ---------------------------------------
# 风控
def a_riskDefend(account):
    condition = 0
    # 涨幅
    quote_rate = history('000001.SH', ['quote_rate'],
                         10, '1d', False, 'pre', is_panel=1)['quote_rate']

    if len(quote_rate)>1:
        if quote_rate[-1]<-3:
            condition += 1
        if quote_rate[-1]<-2.5 and quote_rate[-2]<-2.5:
            condition += 1

    # 停牌判断
    suspension = 0
    for stock in (get_all_securities('stock', get_date()).index):
        #这里把原来的1d改成了1m，从取昨天的变成取今天的了
        paused = history(stock, ['is_paused'], 5, '1d', False,'pre', is_panel=1)['is_paused']
        if len(paused)>0 and paused[0]==1:
            suspension+=1
    if suspension > 20:
        #log.info("停牌条件风控")
        condition += 1
    temp=[]
    if condition > 0:
        for stock in account.holdSl:
            if stock in account.hold:
                trade_target(stock,0)
                temp.append(stock)
                log.info("波段风控卖出"+stock)
    #重置account.holdSl
    account.holdSl = list(set(account.holdSl) ^ set(temp))
# =======================
# 策略B（hitBoard）
# ---------------------------------------
# 买入判断
def b_buyCheck(stock,data):
    v=data.current(stock)[stock]
    close=v.prev_close
    begin=v.open

    gains=(begin-close)/close
    if gains>0.06 or gains<-0.04:
        return -1
    elif gains>=-0.04 and gains<=0.02:
        return 0
    elif gains>0.02 and gains<=0.04:
        return 1
    return 0

# 流通股本
def b_getShareCapital(stock):
    q = query(
        factor.date,
        factor.circulating_cap
    ).filter(
        factor.symbol == stock,
        factor.date == get_date()
    )
    if len(get_factors(q)['factor_circulating_cap'])==0:
        return 0
    elif get_factors(q)['factor_circulating_cap'][0] is None:
        return 0
    else:
        return get_factors(q)['factor_circulating_cap'][0]
# ---------------------------------------
# 卖出判断
def b_sellCheck(stock,data,account):
    if stock in account.hold:
        amount=account.positions[stock].total_amount
        if b_riskDefend(stock,data):
            trade_amount(stock,0)
            log.info("超短风控卖出"+stock)    
            account.holdHb.pop(dictLoc(account.holdHb,stock))
        if b_rsiCheck(stock,data) or b_runtimeTrCheck(stock,data):
            trade_amount(stock,0)
            log.info("超短卖点卖出"+stock)    
            account.holdHb.pop(dictLoc(account.holdHb,stock))
    return
# rsi判断
def b_rsiCheck(stock,data):
    terms=8
    RSI=pd.DataFrame({'rsi1':np.zeros(terms),'rsi2':np.zeros(terms)})

    for i in range(terms):
        rsi1=getRSI(b_deltaCalc(getHistory(stock,7,i,'60m'),data))
        rsi2=getRSI(b_deltaCalc(getHistory(stock,13,i,'60m'),data))
        RSI.ix[i,'rsi1']=rsi1
        RSI.ix[i,'rsi2']=rsi2

    flag=checkthrough(RSI['rsi1'],RSI['rsi2'])
    if flag==-1:
        return True
    else:
        return False
#换手率
def b_runtimeTrCheck(stock,data):
    #log.info(get_datetime())
    if len(data.history(stock,'turnover_rate',1,'1m',True,None)[stock]['turnover_rate'])==0 or len(data.history(stock,'turnover_rate',1,'1d',True,None)[stock]['turnover_rate'])==0:
        return False
    #前日换手量
    else:
        exc_y=data.history(stock,'turnover_rate',1,'1d',True,None)[stock]['turnover_rate'][0]
        exc_n=data.history(stock,'turnover_rate',1,'1m',True,None)[stock]['turnover_rate'][0]
        if exc_n>1.3*exc_y:
            return True
        else:
            return False

#涨跌量的计算
def b_deltaCalc(close,data):
    df=pd.DataFrame({'close':close,'up':np.zeros(len(close)),'down':np.zeros(len(close))})
    for i in range(len(df)):  
        if i==0:  
            df.ix[i,'up']=0
            df.ix[i,'down']=0
        else:  
            if df.ix[i,'close']-df.ix[i-1,'close']>0:
                df.ix[i,'up']=df.ix[i,'close']-df.ix[i-1,'close']
            if df.ix[i,'close']-df.ix[i-1,'close']<0:
                df.ix[i,'down']=df.ix[i,'close']-df.ix[i-1,'close']
    return df
# ---------------------------------------
# 风控
#单日跌幅4%,双日跌7%
def b_riskDefend(stock, data):
    #条件一：单日跌幅超过4%
    if(len(data.history(stock, 'close', 1, '1d', True, None)[stock]['close']))>0:
        begin = data.history(stock, 'close', 1, '1d', True, None)[stock]['close'][0]
        v=data.current(stock)[stock]
        now=v.open

        if now/begin < 0.96:
            # log.info("b_1d")
            return True
        else:
            #两日7%
            begin = data.history(stock, 'close', 2, '1d', True, None)[stock]['close'][0]
            v=data.current(stock)[stock]
            now=v.open
            if now/begin < 0.93:
                # log.info("b_2d")
                return True
    else:
        return False