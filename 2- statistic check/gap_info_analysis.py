import pandas as pd
import numpy as np
import os


def get_gap_info_table(df, std_multiplier):
    pct_groups = pd.cut(df['trade_pct'] / 100, bins=[0, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 1],
                        labels=['0-2%', '2-5%', '5-7%', '7-10%', '10-15%', '15-20%', '20-30%', '30-100%'])
    
    avg_close_days = df.groupby(pct_groups, observed=True)['num_of_days'].mean()
    gaps_per_group = df.groupby(pct_groups, observed=True)['trade_pct'].count()
    gaps_not_closed = df[df['num_of_days'] == -1].groupby(pct_groups, observed=True)['trade_pct'].count()
    
    gaps_drawdown_per_group = df[(df['num_of_days'] > -1)].groupby(pct_groups, observed=True)['max_drawdown'].mean()

    drawdown_per_group1 = df[df['max_drawdown'] < 2].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group2 = df[(df['max_drawdown'] < 5) & (df['max_drawdown'] >= 2)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group3 = df[(df['max_drawdown'] < 7) & (df['max_drawdown'] >= 5)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group4 = df[(df['max_drawdown'] < 10) & (df['max_drawdown'] >= 7)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group5 = df[(df['max_drawdown'] < 15) & (df['max_drawdown'] >= 10)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group6 = df[(df['max_drawdown'] < 20) & (df['max_drawdown'] >= 15)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group7 = df[(df['max_drawdown'] < 30) & (df['max_drawdown'] >= 20)].groupby(pct_groups, observed=True)['max_drawdown'].count()
    drawdown_per_group8 = df[(df['max_drawdown'] >= 30) & (df['num_of_days'] > -1) ].groupby(pct_groups, observed=True)['max_drawdown'].count()
    
    gaps_Maxdrawdown_per_group = df[(df['num_of_days'] > -1)].groupby(pct_groups, observed=True)['max_drawdown'].max()
    gaps_Maxdrawdown_per_group2 = df[(df['num_of_days'] == -1)].groupby(pct_groups, observed=True)['max_drawdown'].mean()
    gaps_Maxdrawdown_per_group3 = df.groupby(pct_groups, observed=True)['max_drawdown'].max()
   
    info_table = pd.concat([gaps_per_group, gaps_not_closed, 
                            avg_close_days, gaps_drawdown_per_group,
                            drawdown_per_group1, drawdown_per_group2, drawdown_per_group3,drawdown_per_group4, drawdown_per_group5,drawdown_per_group6,drawdown_per_group7,drawdown_per_group8,
                              gaps_Maxdrawdown_per_group,gaps_Maxdrawdown_per_group2,gaps_Maxdrawdown_per_group3], axis=1)
    
    info_table.columns = ['gaps', 'gaps not closed',
                          'Avg days till gap close', 'Avg drawdown of closed gaps',
                          'dd < 2%', '2% <= dd < 5%','5% <= dd < 7%', '7% <= dd < 10%', '10% <= dd < 15%', '15% <= dd < 20%','20% <= dd < 30%',
                          'dd >= 30%', 'Max drawdown','-- Avg drawdown - gaps not closed --','Max drawdown - gaps not closed']
    
    info_table['std_multiplier'] = std_multiplier
    info_table.set_index(['std_multiplier', info_table.index], append=False, inplace=True)
    return info_table

def concat_and_export(prev_info_table, export_name):
    final_info_table = pd.concat(prev_info_table).sort_index(level=[0])
    #calculate the ratio of gaps not closed and put it in a new column after gaps not closed
    final_info_table['gaps not closed ratio'] = (final_info_table['gaps not closed'] / final_info_table['gaps'])*100
    final_info_table['gaps not closed ratio'] = final_info_table['gaps not closed ratio'].fillna(0)
    final_info_table['gaps closed ratio'] = (100 - final_info_table['gaps not closed ratio'])

    pct_cols = ['gaps not closed ratio', 'gaps closed ratio', 'Avg drawdown of closed gaps', 'Max drawdown', 'Max drawdown - gaps not closed', '-- Avg drawdown - gaps not closed --']
    for col in pct_cols:
        final_info_table[col] = final_info_table[col].apply(lambda x: f'{x/100:.2%}')

    float_cols = ['Avg days till gap close']
    for col in float_cols:
        final_info_table[col] = final_info_table[col].apply(lambda x: f'{x:.2f}')

    #place the ratio column after the gaps not closed column in the table
    cols = final_info_table.columns.tolist()
    cols.insert(2, cols.pop(cols.index('gaps closed ratio')))
    cols.insert(3, cols.pop(cols.index('gaps not closed ratio')))
    final_info_table = final_info_table[cols]

    final_info_table.to_excel(f'plots/{export_name}.xlsx')


os.makedirs('plots', exist_ok=True)
# std_multipliers = [i for i in np.arange(0.5, 3.5, 0.5)] -> used to check multiple std multipliers, but we decided to use only 1.0
std_multipliers = [1.0]
info_table = []
info_table_50_days = []

for std_multiplier in std_multipliers:
    df = pd.read_pickle(f'gaps by std\\advanced_gap_info_std_{std_multiplier}.pickle')
    df_50_days = df[ (df['num_of_days'] > -1) & (df['num_of_days'] < 51)]
    info_table.append(get_gap_info_table(df.copy(), std_multiplier))
    info_table_50_days.append(get_gap_info_table(df_50_days.copy(), std_multiplier))

concat_and_export(info_table, 'gap_groups')
concat_and_export(info_table_50_days, 'gap_groups_50_days')
