import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import datetime as dt  

class GapAnalysis:
    def __init__(self, symbol, df, index, gap_pct , type, seven_day_momentum, seven_day_std, prev_gap_date ,rsi, atr, vix_level, prev_ohlc_ratio):
        self.symbol = symbol
        self.start_date = df.loc[index, 'date']
        self.distance_from_prev_gap = (df.loc[index, 'date'] - prev_gap_date).days if prev_gap_date is not None else 0
        self.RSI = rsi
        self.ATR = atr
        self.VIX = vix_level
        self.prev_ohlc_ratio = prev_ohlc_ratio
        self.day = df.loc[index, 'day']
        self.week = df.loc[index, 'week']
        self.month = df.loc[index, 'month']
        self.pct_gap = gap_pct
        self.start_idx = index
        self.type = type
        self.open_price = df.loc[index, 'open']
        self.price_to_close = df.loc[index-1, 'low'] if type == 'long' else df.loc[index-1, 'high']
        self.date_of_close = None
        self.num_of_days = -1
        self.industry = df.loc[index, 'industry']
        self.sub_industry = df.loc[index, 'sub_industry']
        self.gap_closed = False
        self.drawdown = 0.0
        self.seven_day_momentum = seven_day_momentum
        self.seven_day_std = seven_day_std
        gap_volume_change = (df.loc[index-2,'volume'] - df.loc[index-1,'volume']) / df.loc[index-1,'volume']
        self.prev2day_gap_volume_direction = 'Up' if gap_volume_change > 0 else 'Down'
        self.prev2day_gap_volume_change = abs(gap_volume_change)
        self.prev_day_direction = 'Up' if df.loc[index-1, 'close'] > df.loc[index-1, 'open'] else 'Down'
        self.prev_day_momentum = (df.loc[index-1, 'close'] - df.loc[index-1, 'open']) / df.loc[index-1, 'open']
        self.ma_5 = df.loc[index, 'mv_avg_5']
        self.ma_10 = df.loc[index, 'mv_avg_10']
        self.ma_20 = df.loc[index, 'mv_avg_20']
        self.ROC_5 = df.loc[index, 'ROC_5']
        self.ROC_7 = df.loc[index, 'ROC_7']
        self.ROC_14 = df.loc[index, 'ROC_14']


    def calc_y(self, dd_threshold):
        if self.type == 'long':
            if self.drawdown <= dd_threshold: #if drawdown is less than {dd_threshold}% return 1(=long) to perform 'close gap' trade
                return 1
            else:
                best_trade = self.pct_gap if self.pct_gap > self.drawdown  else (-1.0)*self.drawdown
                return 1 if best_trade > 0 else 0
        else:
            if self.drawdown <= dd_threshold: #if drawdown is less than {dd_threshold}% return 0(=short) to perform 'close gap' trade
                return 0
            else:
                best_trade = (-1.0)*self.pct_gap if self.pct_gap > self.drawdown else self.drawdown
                return 1 if best_trade > 0 else 0

    def export_to_catboost(self, dd_thresholds:list):
        data = {
            'symbol': self.symbol,
            'date': self.start_date,
            'gap_pct': self.pct_gap if self.type == 'short' else (-1.0) * self.pct_gap,
            'gap_direction': 'Up' if self.type == 'short' else 'Down',
            'gap_volume_change': self.prev2day_gap_volume_change,
            'gap_volume_direction': self.prev2day_gap_volume_direction,
            'momentum of last 7 days': self.seven_day_momentum,
            'direction of last day': self.prev_day_direction,
            'momentum of last day': self.prev_day_momentum,
            'ROC 5': self.ROC_5,
            'ROC 7': self.ROC_7,
            'ROC 14': self.ROC_14,
            'std of last 7 days': self.seven_day_std,
            'RSI': self.RSI,
            'ATR': self.ATR,
            'VIX': self.VIX,
            'prev OHLC ratio': self.prev_ohlc_ratio,
            'SMA 5': self.ma_5,
            'SMA 10': self.ma_10,
            'SMA 20': self.ma_20,
            'distance_from_prev_gap': self.distance_from_prev_gap,
            'day' : self.day,
            'week' : self.week,
            'month': self.month,
            'industry': self.industry,
            'sub_industry': self.sub_industry}
        for dd_threshold in dd_thresholds:
            data[f'y_{dd_threshold}'] = self.calc_y(dd_threshold)
        return data
    
    def update_and_check_pos(self, index, df):
        """returns True if the best trade is closed, False otherwise"""
        if self.type == 'long':
            return self.update_and_check_pos_long(index, df)
        else:
            return self.update_and_check_pos_short(index, df)

    def update_and_check_pos_long(self, index, df):
        """returns True if the trade succseeded on closing the gap is closed, False otherwise"""

        if self.gap_closed == False:
            self.drawdown = max(self.drawdown, ((self.open_price - df.loc[index,'low']) / self.open_price))

        if df.loc[index,'high'] >= self.price_to_close and self.gap_closed == False: #gap closed
            self.date_of_close = df.loc[index,'date']
            self.num_of_days = index - self.start_idx
            self.gap_closed = True

        return self.gap_closed

    def update_and_check_pos_short(self, index, df):
        """returns True if the trade succseeded on closing the gap is closed, False otherwise"""

        if self.gap_closed == False:
            self.drawdown = max(self.drawdown, ((df.loc[index,'high'] - self.open_price) / self.open_price))
            
        if df.loc[index,'low'] <= self.price_to_close and self.gap_closed == False: #gap closed
            self.date_of_close = df.loc[index,'date']
            self.num_of_days = index - self.start_idx
            self.gap_closed = True

        return self.gap_closed


def calc_gap_down(data:pd.DataFrame, gap_threshold: float = 2.0)-> pd.Series:
    """calculate gaps for long positions towards closing the gap"""
    gap = pd.Series([False] * len(data), index=data.index)
    gap_pct_series = pd.Series([0.0] * len(data), index=data.index)
    for i in range(1, len(data)):
        gap_pct = ((data.loc[i-1,'low'] - data.loc[i,'open']) / data.loc[i-1,'low'])
        gap_pct_series[i] = gap_pct
        if gap_pct < 0:
            gap[i] = False
        else:
            gap[i] = (gap_pct >= (gap_threshold/100))
    return gap, gap_pct_series
    
def calc_gap_up(data:pd.DataFrame, gap_threshold: float = 2.0)-> pd.Series:
    """calculate gaps for short positions towards closing the gap"""
    gap = pd.Series([False] * len(data), index=data.index)
    gap_pct_series = pd.Series([0.0] * len(data), index=data.index)
    for i in range(1, len(data)):
        gap_pct = ((data.loc[i,'open'] - data.loc[i-1,'high']) / data.loc[i-1,'high'])
        gap_pct_series[i] = gap_pct
        if gap_pct < 0:
            gap[i] = False
        else:
            gap[i] = (gap_pct >= (gap_threshold/100))
    return gap, gap_pct_series

def calc_ATR(data:pd.DataFrame, period:int = 14)-> pd.Series:
    """calculate ATR with shift(1) for a given period"""
    return (np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))).rolling(window=period).mean().shift(1)

def calc_RSI(data:pd.DataFrame, period:int = 14)-> pd.Series:
    """calculate RSI with shift(1) for a given period"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1)

def calc_vix_level(vix_df, date):
    """calculate vix level for a given date"""
    return vix_df.loc[vix_df['date'] == date, 'close'].values[0]

def calc_prev_ohlc_ratio(data:pd.DataFrame, index:int):
    """calculate the ratio of the previous day's open to close"""
    return (data.loc[index-1,'close'] - data.loc[index-1,'open']) / (data.loc[index-1,'high'] - data.loc[index-1,'low'])

def calc_mov_avg(data:pd.DataFrame, period:int = 14)-> pd.Series:
    """calculate moving average with shift(1) for a given period"""
    return data['close'].rolling(window=period).mean().shift(1)

def calc_ROC(data:pd.DataFrame, period:int = 14)-> pd.Series:
    """calculate rate of change with shift(1) for a given period"""
    return data['close'].diff(period) / data['close'].shift(period).shift(1)

def calc_gaps(filename, dd_thresholds):
    #--- load data
    df = pd.read_pickle(f'..\\..\\1- data\\wrds 1day Data\\pickels\\{filename}')
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    #--- calc features
    atr = calc_ATR(df)
    rsi = calc_RSI(df)
    vix_df = pd.read_pickle(f'..\\..\\1- data\\wrds 1day Data\\vix.pickle')

    df['day'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month

    df['mv_avg_5'] = calc_mov_avg(df, 5)
    df['mv_avg_10'] = calc_mov_avg(df, 10)
    df['mv_avg_20'] = calc_mov_avg(df, 20)

    df['ROC_5'] = calc_ROC(df, 5)
    df['ROC_7'] = calc_ROC(df, 7)
    df['ROC_14'] = calc_ROC(df, 14)

    gap_down, gap_down_pct = calc_gap_down(df, gap_threshold=2.0)
    gap_up, gap_up_pct = calc_gap_up(df, gap_threshold=2.0)

    df_size = len(df)

    symbol = filename.split('_')[0]

    catboost_data = []

    seven_day_momentum = pd.Series([0.0] * len(df), index=df.index)
    seven_day_momentum = df['close'].rolling(window=7, min_periods=1).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x)==7 else float('nan')).shift(1)

    seven_day_std = pd.Series([0.0] * len(df), index=df.index)
    seven_day_std = df['pct_change'].rolling(window=7).std(ddof=0).shift(1)

    #--- loop over the data, find gaps and save each gap with its features and 'y' value
    features_max_window = 20
    prev_gap_date = None
    for index, row in df.iterrows():
        if index < features_max_window :
            continue

        if gap_down[index]:
            # create a new gap object
            gap = GapAnalysis(symbol, df, index, gap_down_pct[index], 'long', seven_day_momentum[index], seven_day_std[index],prev_gap_date, rsi[index], atr[index], calc_vix_level(vix_df, df.loc[index-1, 'date']),calc_prev_ohlc_ratio(df, index))
            #enter loop to check if the gap is closed
            for i in range(index, df_size):
                if gap.update_and_check_pos(i, df):
                    catboost_data.append(gap.export_to_catboost(dd_thresholds))
                    break
            if gap.gap_closed == False:
                catboost_data.append(gap.export_to_catboost(dd_thresholds))
            prev_gap_date = gap.start_date
            gap = None

        if gap_up[index]:
            # create a new gap object
            gap = GapAnalysis(symbol, df, index, gap_up_pct[index], 'short', seven_day_momentum[index], seven_day_std[index],prev_gap_date, rsi[index], atr[index], calc_vix_level(vix_df, df.loc[index-1, 'date']),calc_prev_ohlc_ratio(df, index))
            #enter loop to check if the gap is closed
            for i in range(index, df_size):
                if gap.update_and_check_pos(i, df):
                    catboost_data.append(gap.export_to_catboost(dd_thresholds))
                    break
            if gap.gap_closed == False:
                catboost_data.append(gap.export_to_catboost(dd_thresholds))
            prev_gap_date = gap.start_date
            gap = None
                
    return catboost_data

if __name__ == '__main__':
    os.makedirs('catBoost Data', exist_ok=True)
    files = [f for f in os.listdir(f'..\\..\\1- data\\wrds 1day Data\\pickels') if f.endswith('fixed.pickle')]
    

    # the next 2 rows was used to test which dd_threshold is suitable for us:
    # dd_thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # y_thresholds = ['y_0.0', 'y_0.01', 'y_0.02', 'y_0.03', 'y_0.04', 'y_0.05', 'y_0.06', 'y_0.07', 'y_0.08', 'y_0.09', 'y_0.1']

    # we chose 0.05 is the suitable one
    dd_thresholds = [0.05]
    y_thresholds = ['y_0.05']
    with ProcessPoolExecutor() as executor:
        catboost_data = []
        futures = [executor.submit(calc_gaps, filename, dd_thresholds) for filename in files]
        counter = 1

        for future in as_completed(futures):
            try:
                catboost_data_stock = future.result()
                for catboost in catboost_data_stock:
                    catboost_data.append(catboost)
                print(f'Processed stock {counter}/{len(files)}')
            except Exception as e:
                print(f'Error processing stock: {e}')
            counter += 1    

    #save data per dd_threshold
    catboost_data_df = pd.DataFrame(catboost_data)
    counter = 0
    for y_threshold in y_thresholds:
        dd_threshold = dd_thresholds[counter]
        data_for_y = catboost_data_df.copy()
        #drop the other y columns
        y_thresholds_copy = y_thresholds.copy()
        y_thresholds_copy.remove(y_threshold)
        data_for_y.drop(columns=y_thresholds_copy, inplace=True)
        data_for_y.rename(columns={y_threshold: 'y'}, inplace=True)
        data_for_y.to_excel(f'catBoost Data\\catboost_data{dd_threshold}.xlsx',index=False)
        data_for_y.to_pickle(f'catBoost Data\\catboost_data{dd_threshold}.pickle')
        counter += 1

