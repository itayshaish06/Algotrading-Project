import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

def get_logger(symbol):
    curr_time = pd.Timestamp.now().strftime('%H-%M')
    curr_time = curr_time.replace('-', '')
    curr_date = pd.Timestamp.now().strftime('%d-%m-%Y')

    # create a new file for the logger
    log_file = f'logs\\stocks\\{curr_date}_{curr_time}_{symbol}.log'

    # create logger
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handler to the logger
    if not logger.handlers:
        logger.addHandler(fh)

    return logger

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def fix_open_close(data: pd.DataFrame, symbol: str, logger: logging.Logger):
    ret = False
    for index, row in data.iterrows():
        try:
            if pd.isna(row['close']):
                logger.error(f'{symbol} - Close price is missing in index - {index}')
                raise Exception(f'Close price is missing')
            if pd.isna(row['open']):
                for filename in os.listdir(f'..\\yfinance 1day Data'):
                    if filename.startswith(symbol) and filename.endswith('.pickle') and 'fixed' not in filename:
                        y_finance_df = pd.read_pickle(f'..\\yfinance 1day Data\\{filename}')
                        bad_date = row['date']
                        y_finance_row = y_finance_df[y_finance_df['date'] == bad_date]
                        if y_finance_row.empty:
                            data.at[index, 'open'] = (data.at[index, 'high'] + data.at[index, 'low']) / 2
                            logger.info(f'{symbol} - fixed open price for {bad_date} with average of high and low prices')
                            break
                        else:
                            new_price = float(y_finance_row.iloc[0]['open'])
                            new_price = new_price * row['splits_adjustment_factor']
                            data.at[index, 'open'] = new_price
                            logger.info(f'{symbol} - fixed open price for {bad_date} to using yfinance data')
                            # print(f'Fixed open price for {symbol} on {bad_date} to {new_price}')
                            ret = True
            if row['close'] < 0:
                data.at[index, 'close'] = -1 * row['close']
                ret = True
            if row['open'] < 0:
                data.at[index, 'open'] = -1 * row['open']
                ret = True
        except Exception as e:
            raise Exception(f'Error fixing open/close prices for {symbol} in index - {index}: {e}')
    return ret

def fix_prices(df:pd.DataFrame, symbol:str, logger: logging.Logger):
    try:
        # ----------- Step 1 - calculate 'adjusted close' price -----------
        multiplier = 1
        adjusted_close = np.zeros(len(df))
        dividend = 0
        for index in reversed(range(len(df))):
            row = df.iloc[index]
            if dividend > 0:
                multiplier = multiplier * (1 - (dividend / row['close']))
                dividend = 0
            adjusted_close[index] = multiplier * (row['close'] / row['splits_adjustment_factor'])
            if row['dividend_amount'] is not None:
                if float(row['dividend_amount']) > 0:
                    dividend = row['dividend_amount']
        df['adjusted_close'] = adjusted_close

        # ----------- Step 2 - calculate 'adjusted open', 'adjusted high', 'adjusted low' prices -----------
        df['adjusted_open'] = df['open'] * df['adjusted_close'] / df['close']
        df['adjusted_high'] = df['high'] * df['adjusted_close'] / df['close']
        df['adjusted_low'] = df['low'] * df['adjusted_close'] / df['close']

        keep_columns = ['date', 'adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'volume', 'industry','symbol','sub_industry']
        df = df[keep_columns]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'industry','symbol','sub_industry']
        return df
    except Exception as e:
        logger.error(f'{symbol} - Error fixing prices: {e}')
        raise Exception(f'Error fixing prices for {symbol}: {e}')

def check_continues(df: pd.DataFrame, symbol: str, logger: logging.Logger) -> pd.DataFrame:
    starting_date = df.at[0, 'date']
    bad_days_gap = 5 # 5 was chosen because 1-4 days was legit missing data
    for i in range(1, len(df)):
        if (df.at[i, 'date'] - df.at[i-1, 'date']).days > bad_days_gap:
            logger.error(f'{symbol} - Missing dates between {df.at[i-1, "date"]} and {df.at[i, "date"]}')
            logger.info(f'{symbol} - starting from {df.at[i, "date"]}')
            starting_date = df.at[i, 'date']
    df = df[df['date'] >= starting_date]
    df.reset_index(drop=True, inplace=True)
    return df

def calc_std(df: pd.DataFrame, symbol: str, logger: logging.Logger):
    try:
        if df.at[0, 'close'] == 0 or df.at[0, 'close'] is None:
            raise Exception('Close price is missing')
        df['pct_change'] = df['close'].pct_change()
        df['252d_std'] = df['pct_change'].shift(1).rolling(window=252).std(ddof=0)
        df['30d_std'] = df['pct_change'].shift(1).rolling(window=30).std(ddof=0)
        # fill na values with 0
        df['252d_std'] = df['252d_std'].fillna(0)
        df['30d_std'] = df['30d_std'].fillna(0)
    except Exception as e:
        logger.error(f'{symbol} - Error calculating std: {e}')
        raise Exception(f'Error calculating std for {symbol}: {e}')
    return df

def fix_duplicated_dates(df: pd.DataFrame, symbol: str, duplicates: list, logger: logging.Logger):
    # for pairs of duplicates, consider 2 cases:
    # 1. split adjustment factor is the same for both rows - keep the row with 'dividend_amount' not nan/None or sum the dividend_amounts
    # 2. split adjustment factor is different - error
    to_be_dropped = []
    for i in range(0, len(duplicates) - 1, 2):
        idx = duplicates[i]
        if df.loc[idx, 'splits_adjustment_factor'] == df.loc[idx+1, 'splits_adjustment_factor']:
            if pd.isna(df.loc[idx, 'dividend_amount']) and pd.isna(df.loc[idx+1, 'dividend_amount']):
                to_be_dropped.append(idx+1)
                logger.info(f'{symbol} - {df.loc[idx, "date"]} - Both rows have nan dividend_amount - keeping the first row')
            elif pd.isna(df.loc[idx, 'dividend_amount']) and pd.notna(df.loc[idx+1, 'dividend_amount']):
                to_be_dropped.append(idx)
                logger.info(f'{symbol} - {df.loc[idx, "date"]} - Keeping the row with dividend_amount')
            elif pd.notna(df.loc[idx, 'dividend_amount']) and pd.isna(df.loc[idx+1, 'dividend_amount']):
                to_be_dropped.append(idx+1)
                logger.info(f'{symbol} - {df.loc[idx, "date"]} - Keeping the row with dividend_amount')
            else:
                # dividend_amount is not nan for both rows
                sum_dividends = df.loc[idx, 'dividend_amount'] + df.loc[idx+1, 'dividend_amount']
                df.loc[idx, 'dividend_amount'] = sum_dividends
                to_be_dropped.append(idx+1)
                logger.info(f'{symbol} - {df.loc[idx, "date"]} - Summing dividend_amounts')
        else:
            logger.error(f'{symbol} - {df.loc[idx, "date"]} - Split adjustment factor is different for the duplicated rows')
            df.to_csv(f'{symbol}_duplicated.csv')
    
    df.drop(to_be_dropped, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def process_stock(filename, bad_stocks,sector_df):
        symbol = filename.split('_')[0]
        if filename.endswith('.pickle') and 'fixed' not in filename and symbol not in bad_stocks:
            try :

                logger = get_logger(symbol)
                logger.info(f' -------------------- {symbol} - Starting data fix -------------------- ')
                df = pd.read_pickle(f'original\\{filename}')
                sector = sector_df[sector_df['Symbol'] == symbol]['GICS Sector'].values[0]
                sub_industry = sector_df[sector_df['Symbol'] == symbol]['GICS Sub-Industry'].values[0]

                df['industry'] = sector
                df['sub_industry'] = sub_industry

                # -------------------------------------- Part 1 - insanity checks --------------------------------------
                if df.empty:
                    logger.error(f'{symbol} - Empty table')
                    return f' ---------------------- {symbol} done with error - Empty table'
                
                if df['industry'].unique().size > 1:
                    logger.error(f'{symbol} - More than 1 industry' - {df["industry"].unique()})
                    return f' ---------------------- {symbol} done with error - {df["industry"].unique()}'
                
                df.columns = df.columns.str.lower()

                # -------------------------------------- Part 2 - handle negative values and duplicated rows --------------------------------------
                # go over numeric columns and apply abs
                for col in ['open', 'high', 'low', 'close']:
                    if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                        df[col] = df[col].abs()

                df.drop_duplicates(inplace=True) #remove rows that are identical
                df = df.sort_values(by='date')
                df.reset_index(drop=True, inplace=True)

                # handle rows that has same 'date'
                duplicates = df[df.duplicated(subset='date', keep=False)].index

                if len(duplicates) > 0:
                    df = fix_duplicated_dates(df, symbol, duplicates, logger)

                duplicates = df[df.duplicated(subset='date', keep=False)].index
                if len(duplicates) > 0:
                    logger.error(f'{symbol} - Duplicated dates are not in pairs')
                    return f' ---------------------- {symbol} done with error - Duplicated dates are not in pairs'              
                
                # -------------------------------------- Part 3 - complete missing open prices --------------------------------------
                df['dividend_amount'] = df['dividend_amount'].fillna(0)

                res = fix_open_close(df, symbol, logger)

                # -------------------------------------- Part 4 - fix prices according to splits and dividends --------------------------------------
                df = fix_prices(df, symbol, logger)
                df = check_continues(df, symbol, logger)
                df = calc_std(df, symbol, logger)

                # -------------------------------------- Part 5 - save the fixed data --------------------------------------
                start_date = df.loc[0, 'date'].strftime('%Y-%m-%d')
                end_date = df.loc[len(df)-1, 'date'].strftime('%Y-%m-%d')
                df.to_pickle(f'pickels\\{symbol}_from_{start_date}_to_{end_date}_fixed.pickle')
                df.loc[:, 'date'] = df['date'].astype(str)
                df.to_excel(f'EXELS\\{symbol}_from_{start_date}_to_{end_date}_fixed.xlsx')
                logger.info(f' -------------------- {symbol} - Fixed data saved successfully -------------------- \n')
                #close the logger
                close_logger(logger)
                return f'{symbol} done'

            except Exception as e:
                return f'{symbol} done with error - {e}'

if __name__ == '__main__':
    os.makedirs('EXELS', exist_ok=True)
    os.makedirs('pickels', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs\\stocks', exist_ok=True)
    curr_time = pd.Timestamp.now().strftime('%H-%M')
    curr_time = curr_time.replace('-', '')
    curr_date = pd.Timestamp.now().strftime('%d-%m-%Y')

    # create a new file for the loger
    log_file = f'logs\\{curr_date}_{curr_time}.log'

    with open(f'logs\\results.txt', 'w') as file:
        # -- problem stocks --
        empty_table_stocks = ['BF.B', 'BRK.B', 'CPAY', 'GEV', 'SOLV', 'WELL']
        big_missing_data_stocks = ['BIIB','BIO','AAL','C','DXCM','MKC','SMCI','STZ', 'TAP', 'TFC','VRTX']
        small_info = ['DAY']
        duplicate_not_in_pairs = ['BX','BKR','CB','COR','CTRA','EXPE','GRMN','GEN','HST','JCI', 'L', 'LEN', 'PARA','PLD','SCHW','STLD','TMUS', 'TT']
        symbols_of_more_than_1_industry_from_wrds = ['MGM', 'T','CZR','ES']
        has_negative_price_after_fix = ['PPG']
        bad_stocks = empty_table_stocks + big_missing_data_stocks + small_info + duplicate_not_in_pairs + symbols_of_more_than_1_industry_from_wrds + has_negative_price_after_fix
        
        files = [f for f in os.listdir('original') if f.endswith('.pickle') and 'fixed' not in f and f.split('_')[0] not in bad_stocks]
        sector_df = pd.read_csv('sectors.csv')

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_stock, filename, bad_stocks,sector_df.copy()) for filename in files]
            counter = 1
            for future in as_completed(futures):
                try:
                    file.write(f'{counter}.{future.result()}\n')
                    print(f'{counter}.{future.result()}')
                except Exception as e:
                    print(f'Error processing stock: {e}')
                counter += 1

        with open(log_file, 'w') as file:
            for log in os.listdir('logs\\stocks'):
                if log.endswith('.log'):
                    with open(f'logs\\stocks\\{log}', 'r') as f:
                        file.write(f.read())
                    os.remove(f'logs\\stocks\\{log}')