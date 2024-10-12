import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class GapAnalysis:
    def __init__(self, symbol, start_date, std, start_idx, type, open_price, price_to_close, high, industry, sub_industry):
        self.symbol = symbol
        self.start_date = start_date
        self.std = std
        self.pct_gap = ((price_to_close - open_price) / price_to_close)*100 if type == 'long' else ((open_price - price_to_close) / price_to_close)*100
        self.start_idx = start_idx
        self.type = type
        self.open_price = open_price
        self.price_to_close = price_to_close
        self.date_of_close = None
        self.num_of_days = -1
        self.industry = industry
        self.sub_industry = sub_industry
        self.gap_closed = False
        self.trade_pct = ((price_to_close - open_price) / open_price)*100 if type == 'long' else ((open_price - price_to_close) / open_price)*100
        self.drawdown = 0.0

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'pct_gap': self.pct_gap,
            'std': self.std*100,
            'type': self.type,
            'open_price': self.open_price,
            'price_to_close': self.price_to_close,
            'date_of_close': self.date_of_close,
            'num_of_days': self.num_of_days,
            'industry': self.industry,
            'sub_industry': self.sub_industry,
            'gap_closed': self.gap_closed,
            'trade_pct': self.trade_pct,
            'max_drawdown': self.drawdown
        }
    
    def update_and_check_pos(self, index, df):
        """returns True if the best trade is closed, False otherwise"""
        if self.type == 'long':
            return self.update_and_check_pos_long(index, df)
        else:
            return self.update_and_check_pos_short(index, df)
 
    def update_and_check_pos_long(self, index, df):
        """returns True if the Gap Closed, False otherwise"""
        if df.loc[index,'high'] >= self.price_to_close and self.gap_closed == False: #gap closed
            self.date_of_close = df.loc[index,'date']
            self.num_of_days = index - self.start_idx
            self.gap_closed = True
            self.trade_pct = ((self.price_to_close - self.open_price) / self.open_price)*100
        self.drawdown = max(self.drawdown, ((self.open_price - df.loc[index,'low']) / self.open_price)*100)
        return self.gap_closed

    def update_and_check_pos_short(self, index, df):
        """returns True if the Gap Closed, False otherwise"""
        if df.loc[index,'low'] <= self.price_to_close and self.gap_closed == False: #gap closed
            self.date_of_close = df.loc[index,'date']
            self.num_of_days = index - self.start_idx
            self.gap_closed = True
            self.trade_pct = ((self.open_price - self.price_to_close) / self.open_price)*100
        self.drawdown = max(self.drawdown, ((df.loc[index,'high'] - self.open_price) / self.open_price)*100)
        return self.gap_closed
      
def calc_gap_down(data:pd.DataFrame, std_multiplier: float, gap_threshold: float = 2.0)-> pd.Series:
    gap = pd.Series([False] * len(data), index=data.index)
    prev_gap = None
    for i in range(1, len(data)):
        if data.loc[i,'252d_std'] == 0 or pd.isna(data.loc[i,'252d_std']):
            gap[i] = False
            continue
        gap_pct = ((data.loc[i-1,'low'] - data.loc[i,'open']) / data.loc[i-1,'low'])
        if gap_pct < 0:
            gap[i] = False
            continue
        gap[i] = (gap_pct >= std_multiplier*(data.loc[i,'252d_std'])) and (gap_pct >= (gap_threshold/100))
        gap_pct = None
    return gap
    
def calc_gap_up(data:pd.DataFrame, std_multiplier: float, gap_threshold: float = 2.0)-> pd.Series:
    gap = pd.Series([False] * len(data), index=data.index)
    for i in range(1, len(data)):
        if data.loc[i,'252d_std'] == 0 or pd.isna(data.loc[i,'252d_std']):
            gap[i] = False
            continue
        gap_pct = ((data.loc[i,'open'] - data.loc[i-1,'high']) / data.loc[i-1,'high'])
        if gap_pct < 0:
            gap[i] = False
            continue
        gap[i] = (gap_pct >= std_multiplier*(data.loc[i,'252d_std'])) and (gap_pct >= (gap_threshold/100))
        gap_pct = None
    return gap

def calc_gaps(filename, std_multiplier: float):
    df = pd.read_pickle(f'..\\1- data\\wrds 1day Data\\pickels\\{filename}')

    df.columns = df.columns.str.lower()

    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)

    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)

    threshold_start_date = '2013-01-01'
    threshold_end_date = '2025-01-01'

    df = df[(df.date >= threshold_start_date) & (df.date < threshold_end_date)]

    df = df.sort_values(by='date').reset_index(drop=True)

    df['252d_std'].fillna(0, inplace=True)

    if df['252d_std'].isnull().values.any():
        print('Error: NaN in std')
        return []

    gap_down = calc_gap_down(df, std_multiplier,gap_threshold=0.0)
    gap_up = calc_gap_up(df, std_multiplier,gap_threshold=0.0)

    df_size = len(df)

    symbol = filename.split('_')[0]

    gap_info = []

    for index, row in df.iterrows():
        if gap_down[index]:
            # enter long position in order to check if the gap closes
            gap = GapAnalysis(symbol, df.loc[index, 'date'], df.loc[index,'252d_std'], index, 'long', df.loc[index, 'open'], df.loc[index-1, 'low'], df.loc[index, 'high'], df.loc[index, 'industry'], df.loc[index, 'sub_industry'])
            for i in range(index, df_size):
                if gap.update_and_check_pos(i, df):
                    gap_info.append(gap.to_dict())
                    break
            if gap.gap_closed == False:
                gap_info.append(gap.to_dict())

        if gap_up[index]:
            # enter short position in order to check if the gap closes
            gap = GapAnalysis(symbol, df.loc[index, 'date'], df.loc[index,'252d_std'], index, 'short', df.loc[index, 'open'], df.loc[index-1, 'high'], df.loc[index, 'low'], df.loc[index, 'industry'], df.loc[index, 'sub_industry'])
            for i in range(index, df_size):
                if gap.update_and_check_pos(i, df):
                    gap_info.append(gap.to_dict())
                    break
            if gap.gap_closed == False:
                gap_info.append(gap.to_dict())

        if gap_up[index] and gap_down[index]:
            print(f'True on both gap calculation -> {symbol} on {df.loc[index, "date"]}')

    return gap_info

if __name__ == '__main__':
    os.makedirs('gaps by std', exist_ok=True)

    # std_multipliers = [i for i in np.arange(0.5, 3.5, 0.5)] # -> used for std analysis -> but we decided to use only 1.0
    std_multipliers = [1.0]
    stock_counter = 1
    files = [f for f in os.listdir(f'..\\1- data\\wrds 1day Data\\pickels') if f.endswith('fixed.pickle')]

    for std_multiplier in std_multipliers:
        with ProcessPoolExecutor() as executor:
            gap_info = []
            futures = [executor.submit(calc_gaps, filename, std_multiplier) for filename in files]
            counter = 1
            for future in as_completed(futures):
                try:
                    for gap in future.result():
                        gap_info.append(gap)
                    print(f'Processed stock {counter}/{len(files)}, std_multiplier: {std_multiplier}')
                except Exception as e:
                    print(f'Error processing stock: {e}')
                counter += 1    
        gap_info_df = pd.DataFrame(gap_info)
        gap_info_df.to_excel(f'gaps by std/advanced_gap_info_std_{std_multiplier}.xlsx')
        gap_info_df.to_pickle(f'gaps by std/advanced_gap_info_std_{std_multiplier}.pickle')
        gap_info = None



