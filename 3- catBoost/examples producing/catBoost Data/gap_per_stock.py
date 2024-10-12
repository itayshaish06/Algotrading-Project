import pandas as pd
import numpy as np


# ----------------- load data -----------------

data = pd.read_pickle('catboost_data0.05.pickle')
data = data.dropna()
data = data.drop_duplicates()
data = data.reset_index(drop=True)

# ----------------- calc gaps per year -----------------

# filter out the rows with date after 2020-01-01 (in sample is before 2020-01-01)
data['date'] = pd.to_datetime(data['date'])
data = data[data['date'] < '2020-01-01']

#add a year column
data['year'] = data['date'].dt.year

# gruop by symbol
data_grouped = data.groupby('symbol')

# get the number of gaps for each symbol per year
gaps_per_2005 = data_grouped.apply(lambda x: x[x['year'] == 2005]['gap_pct'].count())
gaps_per_2006 = data_grouped.apply(lambda x: x[x['year'] == 2006]['gap_pct'].count())
gaps_per_2007 = data_grouped.apply(lambda x: x[x['year'] == 2007]['gap_pct'].count())
gaps_per_2008 = data_grouped.apply(lambda x: x[x['year'] == 2008]['gap_pct'].count())
gaps_per_2009 = data_grouped.apply(lambda x: x[x['year'] == 2009]['gap_pct'].count())
gaps_per_2010 = data_grouped.apply(lambda x: x[x['year'] == 2010]['gap_pct'].count())
gaps_per_2011 = data_grouped.apply(lambda x: x[x['year'] == 2011]['gap_pct'].count())
gaps_per_2012 = data_grouped.apply(lambda x: x[x['year'] == 2012]['gap_pct'].count())
gaps_per_2013 = data_grouped.apply(lambda x: x[x['year'] == 2013]['gap_pct'].count())
gaps_per_2014 = data_grouped.apply(lambda x: x[x['year'] == 2014]['gap_pct'].count())
gaps_per_2015 = data_grouped.apply(lambda x: x[x['year'] == 2015]['gap_pct'].count())
gaps_per_2016 = data_grouped.apply(lambda x: x[x['year'] == 2016]['gap_pct'].count())
gaps_per_2017 = data_grouped.apply(lambda x: x[x['year'] == 2017]['gap_pct'].count())
gaps_per_2018 = data_grouped.apply(lambda x: x[x['year'] == 2018]['gap_pct'].count())
gaps_per_2019 = data_grouped.apply(lambda x: x[x['year'] == 2019]['gap_pct'].count())

# get the number of gaps for each symbol
gaps = data_grouped['gap_pct'].count()

# get the number of gaps for each symbol from 2012 to 2019
gaps_per_2012_2019 = data_grouped.apply(lambda x: x[x['year'] >= 2012]['gap_pct'].count())

# get the number of gaps for each symbol per year
gaps_per_year = pd.DataFrame({'2005': gaps_per_2005, '2006': gaps_per_2006, '2007': gaps_per_2007, '2008': gaps_per_2008, '2009': gaps_per_2009, '2010': gaps_per_2010, '2011': gaps_per_2011, '2012': gaps_per_2012, '2013': gaps_per_2013, '2014': gaps_per_2014, '2015': gaps_per_2015, '2016': gaps_per_2016, '2017': gaps_per_2017, '2018': gaps_per_2018, '2019': gaps_per_2019, 'total 2012-2019': gaps_per_2012_2019, 'total': gaps})

# export the data
gaps_per_year.to_csv('gaps_per_year.csv')

# ----------------- calc top 100 stocks -----------------

# sort gaps_per_year according to 'total 2012-2019' from high to low
#export the top 100 symbols
gaps_per_year = gaps_per_year.sort_values(by='total 2012-2019', ascending=False)
top_100 = gaps_per_year.head(100)
print(top_100.columns)
top_100.reset_index(inplace=True)
print(top_100.columns)
top_100 = top_100.rename(columns={"symbol": "backtesting stocks"})
top_100
top_100['backtesting stocks'].to_excel('backtest_stocks.xlsx', index=False)
