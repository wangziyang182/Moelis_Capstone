import logging as logging
import numpy as np
import pandas as pd

from scipy.stats import mstats

logger = logging.getLogger(__name__)


dict_manual_renamings = {
    'campaign_symbol': 'campaign_id',
    'activist': 'activist_name',
    'campaign_announce_date': 'campaign_announcement_date',
    'in_force_prior_to_announcement_poison_pill': 'poison_pill_in_force_prior_to_announcement',
    'adopted_in_response_to_campaign_poison_pill': 'poison_pill_adopted_in_response_to_campaign'
}

list_drop_columns = [
    
    # repeat of campaign_announce_date
    'announcement_date_date'
    
]

list_column_order = [
    
    # campaign
    'campaign_id',
    'campaign_announcement_date',
    'campaign_title',
    'campaign_objective_primary',
    'value_demand',
    'governance_demand',
    'activist_campaign_tactic',
    'total_number_of_board_seats',
    'number_of_board_seats_sought',
    'short_or_majority_or_full_slate',
    'proxy_proposal',
    'glass_lewis_support',
    'iss_support',
    
    # activist
    'activist_id',
    'activist_name',
    'activist_group',
    'first_trade_date',
    'last_trade_date',
    'ownership_pecent_on_announcement',
    
    # company
    'company_id',
    'company_name',
    'sector',
    'price_at_announcement',
    'ltm_eps_at_announcement',
    'earnings_yield_at_announcement',
    'current_entity_status',
    'current_entity_detail',
    'public_before_or_after_campaign_announcement',
    'poison_pill_in_force_prior_to_announcement',
    'poison_pill_adopted_in_response_to_campaign',
    'pre_18m_announcement_date',
    'pre_12m_announcement_date',
    'pre_6m_announcement_date',
    'pre_3m_announcement_date',
    'pre_18m_stock_price',
    'pre_12m_stock_price',
    'pre_6m_stock_price',
    'pre_3m_stock_price',
    'pre_18m_price_to_earnings',
    'pre_12m_price_to_earnings',
    'pre_6m_price_to_earnings',
    'pre_3m_price_to_earnings',    
    'pre_18m_dividends',
    'pre_12m_dividends',
    'pre_6m_dividends',
    'pre_3m_dividends',
    'pre_18m_price_return',
    'pre_12m_price_return',
    'pre_6m_price_return',
    'pre_3m_price_return',
    'pre_18m_total_return',
    'pre_12m_total_return',
    'pre_6m_total_return',
    'pre_3m_total_return',

    # campaign success outcomes
    'number_of_board_seats_gained', 
    'proxy_campaign_winner_or_result',
    'activist_campaign_results',

    # return success outcomes
    'post_6m_announcement_date',
    'post_12m_announcement_date',
    'post_18m_announcement_date',    
    'post_6m_stock_price',
    'post_12m_stock_price',
    'post_18m_stock_price',
    'post_6m_price_to_earnings',
    'post_12m_price_to_earnings',
    'post_18m_price_to_earnings',
    'post_6m_dividends',
    'post_12m_dividends',
    'post_18m_dividends',
    'post_6m_price_return',
    'post_12m_price_return',
    'post_18m_price_return',
    'post_6m_total_return',  
    'post_12m_total_return', 
    'post_18m_total_return'

]

def clean_column_name(column_name):
    return (
        column_name
        .strip()
        .lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('\n', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('%', 'pecent')
        .replace('/', '_or_')
        .replace('?', '')
        .replace('__', '_')
        .replace('_dates', '_date')
        .replace('_announcements', '_announcement')
        .replace('pre_date', 'pre')
        .replace('post_date', 'post')
        .replace('18_months_pre', 'pre_18m')
        .replace('1_year_pre', 'pre_12m')
        .replace('6_months_pre', 'pre_6m')
        .replace('90_days_pre', 'pre_3m')
        .replace('18_months_post', 'post_18m')
        .replace('1_year_post', 'post_12m')
        .replace('6_months_post', 'post_6m')
        .replace('90_days_post', 'post_3m')        
    )


def read_factset_campaign_data():
    logger.info('reading factset campaign data')
    df_factset_campaign = pd.read_excel('data/FactSet_Campaign v9.xlsx', skiprows=2, na_values=['', ' ', '_', '-', ' - ', '- ', ' -', 'na', 'NA', 'n.a.', '#NAME?'])
    return df_factset_campaign

def read_factset_pricing_data():

    logger.info('reading factset pricing data')
    df_factset_pricing = pd.read_csv('data/FactSet_Pricing.txt', parse_dates=['FSDate'])

    df_factset_pricing = (
        df_factset_pricing
        .rename(columns={
            'FactSetID': 'company_id',
            'FSDate': 'date',
            'FGPRICE': 'price',
            'FGVolume': 'volume'
        })
        .sort_values(['company_id', 'date'])
        .assign(price=lambda df: df.price.astype(float))
        .assign(stock_daily_return=lambda df: df.groupby('company_id').price.pct_change())
        .assign(stock_daily_return=lambda df: df.stock_daily_return.clip(-0.50, 0.50))
    )

    return df_factset_pricing


def read_sp_pricing_data():

    logger.info('reading sp pricing data')
    df_sp_pricing = pd.read_csv('data/sp.csv', parse_dates=['Date'])

    df_sp_pricing = (
        df_sp_pricing
        .loc[:, ['Date', 'Adj Close']]
        .rename(columns={
            'Date': 'date',
            'Adj Close': 'price',
        })
        .sort_values(['date'])
        .assign(sp_daily_return=lambda df: df.price.pct_change())
    )

    return df_sp_pricing


def clean_data(df_factset_campaign):

    df_cleaning = df_factset_campaign.copy()

    logger.info('renaming and dropping columns')
    df_cleaning = (
        df_cleaning
        .rename(columns=clean_column_name)
        .rename(columns=dict_manual_renamings)
        .drop(axis='columns', labels=list_drop_columns)
        .reindex(columns=list_column_order)
    )

    logger.info('formatting date columns')
    df_cleaning = (
        df_cleaning
        # convert strings to dates based on format
        # note this fails silently for malformed dates for now
        .assign(campaign_announcement_date=lambda df: pd.to_datetime(df.campaign_announcement_date.astype(str), infer_datetime_format=True))
        .assign(first_trade_date=lambda df: pd.to_datetime(df.first_trade_date.astype(str), infer_datetime_format=True))
        .assign(last_trade_date=lambda df: pd.to_datetime(df.last_trade_date.astype(str), infer_datetime_format=True, errors='coerce'))
    )

    logger.info('formatting string columns')
    df_cleaning = (
        df_cleaning
        # extract company name and activist group from campaign title
        # note that what comes after the / can be a list of comma separated activist names, I call this activist group
        .assign(company_name=lambda df: df.campaign_title.str.split(' / ', n=1, expand=True)[0])
        .assign(activist_group=lambda df: df.campaign_title.str.split(' / ', n=1, expand=True)[1])
        # for categoricals, standardize to Title case
        .assign(sector=lambda df: df.sector.str.title())
        .assign(public_before_or_after_campaign_announcement=lambda df: df.public_before_or_after_campaign_announcement.str.title())
        .assign(current_entity_status=lambda df: df.current_entity_status.str.title())
        .assign(current_entity_detail=lambda df: df.current_entity_detail.str.title())
    )

    # reorder
    logger.info('sorting rows')
    df_cleaning = (
        df_cleaning
        .sort_values(['campaign_id', 'campaign_announcement_date', 'campaign_title'])
    )
        
    df_factset_campaign_cleaned = df_cleaning

    return df_factset_campaign_cleaned

def calculate_betas(df_factset_campaign, df_factset_pricing, df_sp_pricing):

    logger.info('merging factset pricing and sp pricing')
    df_pricing = (
        df_factset_pricing
        .pipe(pd.merge, df_sp_pricing[['date', 'sp_daily_return']], how='left', on='date')
    )

    logger.info('merging factset pricing and factset campaign')
    df_pricing_relevant_for_campaign = (
        pd.merge(
            df_pricing,
            df_factset_campaign[[
                'campaign_id',
                'company_id',
                'pre_18m_announcement_date',
                'pre_12m_announcement_date',
                'pre_6m_announcement_date',
                'pre_3m_announcement_date',
                'campaign_announcement_date',
                'post_6m_announcement_date',
                'post_12m_announcement_date',
                'post_18m_announcement_date'
            ]],
            on=['company_id'],
            how='left'
        )
        .loc[lambda df: df.date.between(df['pre_18m_announcement_date'], df['post_18m_announcement_date'])]
    )

    def calculate_beta(gb):
        return pd.Series({
            'pre_18m_announcement_date': gb['pre_18m_announcement_date'].iloc[0],
            'campaign_announcement_date': gb['campaign_announcement_date'].iloc[0],
            'post_18m_announcement_date': gb['post_18m_announcement_date'].iloc[0],
            'pre_18m_market_return': gb.loc[lambda df: df.date.between(df['pre_18m_announcement_date'], df['campaign_announcement_date'])].sp_daily_return.sum(),
            'pre_12m_market_return': gb.loc[lambda df: df.date.between(df['pre_12m_announcement_date'], df['campaign_announcement_date'])].sp_daily_return.sum(),
            'pre_6m_market_return': gb.loc[lambda df: df.date.between(df['pre_6m_announcement_date'], df['campaign_announcement_date'])].sp_daily_return.sum(),
            'post_6m_market_return': gb.loc[lambda df: df.date.between(df['campaign_announcement_date'], df['post_6m_announcement_date'])].sp_daily_return.sum(),
            'post_12m_market_return': gb.loc[lambda df: df.date.between(df['campaign_announcement_date'], df['post_12m_announcement_date'])].sp_daily_return.sum(),
            'post_18m_market_return': gb.loc[lambda df: df.date.between(df['campaign_announcement_date'], df['post_18m_announcement_date'])].sp_daily_return.sum(),
            'beta': gb[['stock_daily_return', 'sp_daily_return']].dropna().cov().iloc[0, 1] / gb[['sp_daily_return']].var().iloc[0]
        })

    df_factset_pricing_beta = df_pricing_relevant_for_campaign.groupby(['campaign_id', 'company_id']).apply(calculate_beta)
    df_factset_pricing_beta['beta'] = df_factset_pricing_beta['beta'].clip(-1, 2)

    return df_factset_pricing_beta

def engineer_features(df_factset_campaign_cleaned, df_factset_betas):

    df_engineering = df_factset_campaign_cleaned.copy()

    # fundamentals
    logger.info('calculating earnings yield')
    df_engineering = (
        df_engineering
        .assign(pre_18m_earnings_yield=lambda df: 1 / df.pre_18m_price_to_earnings)
        .assign(pre_12m_earnings_yield=lambda df: 1 / df.pre_12m_price_to_earnings)
        .assign(pre_6m_earnings_yield=lambda df: 1 / df.pre_6m_price_to_earnings)
        .assign(pre_3m_earnings_yield=lambda df: 1 / df.pre_3m_price_to_earnings)
        .assign(earnings_yield_at_announcement=lambda df: df.ltm_eps_at_announcement / df.price_at_announcement)
        .assign(post_6m_earnings_yield=lambda df: 1 / df.post_6m_price_to_earnings)
        .assign(post_12m_earnings_yield=lambda df: 1 / df.post_12m_price_to_earnings)
        .assign(post_18m_earnings_yield=lambda df: 1 / df.post_18m_price_to_earnings)
    )

    # price returns
    logger.info('calculating price returns')
    df_engineering['pre_18m_price_return'] = df_engineering['price_at_announcement'] / df_engineering['pre_18m_stock_price'] - 1
    df_engineering['pre_12m_price_return'] = df_engineering['price_at_announcement'] / df_engineering['pre_12m_stock_price'] - 1
    df_engineering['pre_6m_price_return'] = df_engineering['price_at_announcement'] / df_engineering['pre_6m_stock_price'] - 1
    df_engineering['pre_3m_price_return'] = df_engineering['price_at_announcement'] / df_engineering['pre_3m_stock_price'] - 1
    df_engineering['post_6m_price_return'] = df_engineering['post_6m_stock_price'] / df_engineering['price_at_announcement'] - 1
    df_engineering['post_12m_price_return'] = df_engineering['post_12m_stock_price'] / df_engineering['price_at_announcement'] - 1
    df_engineering['post_18m_price_return'] = df_engineering['post_18m_stock_price'] / df_engineering['price_at_announcement'] - 1

    # campaign tactics
    # originally provided as a single column with each value being a tuple of tactics
    # explode this tuple into columns, then pivot so each tactic gets one column
    # then create dummies for whether a tactic was used by an activist for a given campaign
    # merge back onto the main data set
    logger.info('calculating campaign tactics')
    df_tactic = (
        df_engineering
        .groupby('campaign_id')
        [
            'activist_id',
            'company_id',
            'activist_campaign_tactic'
        ]
        .last()
        .reset_index()
        .assign(activist_campaign_tactic=lambda df: df.activist_campaign_tactic.fillna('No or Unknown'))
        .assign(activist_campaign_tactic=lambda df: df.activist_campaign_tactic.str.split(', '))
        .explode('activist_campaign_tactic')
        .assign(activist_campaign_tactic_indicator=1)
    )
    df_tactics_indicators = (
        pd.pivot_table(df_tactic, index=['campaign_id'], columns=['activist_campaign_tactic'], values='activist_campaign_tactic_indicator', fill_value=0)
        .rename(columns=clean_column_name)
        .rename(columns=lambda c: 'used_' + c + '_tactic')
    )
    df_engineering = pd.merge(
        df_engineering, df_tactics_indicators,
        on='campaign_id', how='left'
    )

    # past successes
    # create a simple measure of success like whether one year future returns were positive
    # count the cumulative successes by activist
    # make sure to lag our knowledge of successes by more than one year to ensure no look-ahead bias
    # merge back onto the main data set
    logger.info('calculating past successes')
    df_successes = (
        df_factset_campaign_cleaned
        .assign(is_return_success=lambda df: 1 * (
            (df['proxy_campaign_winner_or_result'] == 'Dissident') |
            (df['proxy_campaign_winner_or_result'] == 'Settled/Concessions Made') |
            (df['number_of_board_seats_gained'] > 0)
        ))
        .assign(is_return_success=lambda df: df.is_return_success.fillna(0))
        .loc[:, ['activist_id', 'activist_name', 'campaign_announcement_date', 'is_return_success']]
        .groupby(['activist_id', 'campaign_announcement_date']).last().reset_index()
        .set_index(['activist_id', 'campaign_announcement_date']).sort_index().reset_index()
    )
    df_cumulative_successes = (
        df_successes
        .sort_values(['activist_id', 'campaign_announcement_date'])
        .assign(past_return_successes=lambda df: df.groupby(['activist_id']).is_return_success.cumsum())
        .assign(lagged_campaign_announcement_date=lambda df: df.campaign_announcement_date + pd.offsets.DateOffset(400))
    )
    df_engineering = pd.merge_asof(
        df_engineering.sort_values('campaign_announcement_date'),
        df_cumulative_successes[['activist_id', 'lagged_campaign_announcement_date', 'past_return_successes']].sort_values('lagged_campaign_announcement_date'),
        by=['activist_id'], left_on=['campaign_announcement_date'], right_on=['lagged_campaign_announcement_date']
    )

    # betas
    logger.info('calculating residual returns')
    df_engineering = pd.merge(
        df_engineering,
        df_factset_betas[[
            'campaign_id',
            'company_id',
            'pre_18m_market_return',
            'pre_12m_market_return',
            'pre_6m_market_return',
            'post_6m_market_return',
            'post_12m_market_return',
            'post_18m_market_return',
            'beta',
        ]],
        how='left',
        on=['campaign_id', 'company_id']
    )

    # clean market returns and beta when missing
    for column in df_engineering.columns.tolist():
        if 'market_return' in column:
            df_engineering[column] = df_engineering[column].fillna(0)
        if 'beta' in column:
            df_engineering[column] = df_engineering[column].fillna(0)

    df_engineering['pre_6m_residual_return'] = df_engineering['pre_6m_total_return'] - df_engineering['pre_6m_market_return'] * df_engineering['beta']
    df_engineering['pre_12m_residual_return'] = df_engineering['pre_12m_total_return'] - df_engineering['pre_12m_market_return'] * df_engineering['beta']
    df_engineering['pre_18m_residual_return'] = df_engineering['pre_18m_total_return'] - df_engineering['pre_18m_market_return'] * df_engineering['beta']    
    df_engineering['post_6m_residual_return'] = df_engineering['post_6m_total_return'] - df_engineering['post_6m_market_return'] * df_engineering['beta']
    df_engineering['post_12m_residual_return'] = df_engineering['post_12m_total_return'] - df_engineering['post_12m_market_return'] * df_engineering['beta']
    df_engineering['post_18m_residual_return'] = df_engineering['post_18m_total_return'] - df_engineering['post_18m_market_return'] * df_engineering['beta']

    # coalsece returns
    logger.info('coalescing residual, total and price returns')
    df_engineering['pre_6m_residual_return']   = df_engineering['pre_6m_residual_return'].fillna(df_engineering['pre_6m_total_return'].fillna(df_engineering['pre_6m_price_return']))  
    df_engineering['pre_12m_residual_return']  = df_engineering['pre_12m_residual_return'].fillna(df_engineering['pre_12m_total_return'].fillna(df_engineering['pre_12m_price_return'])) 
    df_engineering['pre_18m_residual_return']  = df_engineering['pre_18m_residual_return'].fillna(df_engineering['pre_18m_total_return'].fillna(df_engineering['pre_18m_price_return'])) 
    df_engineering['post_6m_residual_return']  = df_engineering['post_6m_residual_return'].fillna(df_engineering['post_6m_total_return'].fillna(df_engineering['post_6m_price_return'])) 
    df_engineering['post_12m_residual_return'] = df_engineering['post_12m_residual_return'].fillna(df_engineering['post_12m_total_return'].fillna(df_engineering['post_12m_price_return']))
    df_engineering['post_18m_residual_return'] = df_engineering['post_18m_residual_return'].fillna(df_engineering['post_18m_total_return'].fillna(df_engineering['post_18m_price_return']))

    # annualize
    logger.info('converting all returns to monthly frequency returns')

    df_engineering['pre_18m_price_return'] = df_engineering['pre_18m_price_return'] / 18
    df_engineering['pre_12m_price_return'] = df_engineering['pre_12m_price_return'] / 12
    df_engineering['pre_6m_price_return'] = df_engineering['pre_6m_price_return'] / 6
    df_engineering['pre_3m_price_return'] = df_engineering['pre_3m_price_return'] / 3
    df_engineering['post_6m_price_return'] = df_engineering['post_6m_price_return'] / 3
    df_engineering['post_12m_price_return'] = df_engineering['post_12m_price_return'] / 12
    df_engineering['post_18m_price_return'] = df_engineering['post_18m_price_return'] / 18

    df_engineering['pre_18m_total_return'] = df_engineering['pre_18m_total_return'] / 18
    df_engineering['pre_12m_total_return'] = df_engineering['pre_12m_total_return'] / 12
    df_engineering['pre_6m_total_return'] = df_engineering['pre_6m_total_return'] / 6
    df_engineering['pre_3m_total_return'] = df_engineering['pre_3m_total_return'] / 3
    df_engineering['post_6m_total_return'] = df_engineering['post_6m_total_return'] / 6
    df_engineering['post_12m_total_return'] = df_engineering['post_12m_total_return'] / 12
    df_engineering['post_18m_total_return'] = df_engineering['post_18m_total_return'] / 18

    df_engineering['pre_18m_residual_return'] = df_engineering['pre_18m_residual_return'] / 18
    df_engineering['pre_12m_residual_return'] = df_engineering['pre_12m_residual_return'] / 12
    df_engineering['pre_6m_residual_return'] = df_engineering['pre_6m_residual_return'] / 6
    df_engineering['post_6m_residual_return'] = df_engineering['post_6m_residual_return'] / 6
    df_engineering['post_12m_residual_return'] = df_engineering['post_12m_residual_return'] / 12
    df_engineering['post_18m_residual_return'] = df_engineering['post_18m_residual_return'] / 18

    # winsorize

    logger.info('winsorizing returns')

    # clean market returns and beta when missing
    for column in df_engineering.columns.tolist():
        if 'return' in column:
            df_engineering[column] = df_engineering[column].clip(-1, 2)
        if 'earnings_yield' in column:
            df_engineering[column] = df_engineering[column].clip(-1, 2)
        if 'dividend' in column:
            df_engineering[column] = df_engineering[column].clip(0, 1)

    # for horizon in ['3m', '6m', '12m', '18m']:
    #     for direction in ['pre', 'post']:
    #         for return_type in ['price_return', 'total_return', 'residual_return']:
    #             return_column = f'{direction}_{horizon}_{return_type}_return'
    #             try:
    #                 df_engineering[return_column] = df_engineering[return_column].clip(-1, 2)
    #             except:
    #                 pass

    # cumulative abnormal returns
    logger.info('calculating cumulative abnormal returns')
    df_engineering['cumulative_6m_residual_return'] = df_engineering['pre_6m_residual_return'] + df_engineering['post_6m_residual_return']
    df_engineering['cumulative_12m_residual_return'] = df_engineering['pre_12m_residual_return'] + df_engineering['post_12m_residual_return']
    df_engineering['cumulative_18m_residual_return'] = df_engineering['pre_18m_residual_return'] + df_engineering['post_18m_residual_return']

    # filter
    #logger.info('filtering')
    #df_engineering = (
    #    df_engineering
    #    .loc[lambda df: df.campaign_objective_primary != '13D Filer - No Publicly Disclosed Activism']
    #)

    return df_engineering

def read_clean_table(file_path):

    df = pd.read_csv(file_path)

    for column in df.columns:
        if 'date' in column:
            df[column] = pd.to_datetime(df[column], infer_datetime_format=True)
    
    return df

def write_data(df, write_path):
    df.to_csv(write_path, index=False)
    return

def h1(x):
    return '\n'.join([
        '',
        x.upper(),
        '=' * 120,
        ''
    ])

def main():

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

    if True:

        print(h1('reading'))
        df_factset_campaign = read_factset_campaign_data()
        df_factset_pricing = read_factset_pricing_data()
        df_sp_pricing = read_sp_pricing_data()

        print(h1('cleaning'))
        df_factset_campaign_cleaned = clean_data(df_factset_campaign)
        write_data(df_factset_campaign_cleaned, 'data/clean_factset_campaign_data.csv')
        write_data(df_factset_pricing, 'data/clean_factset_pricing.csv')

        print(h1('betas'))
        df_factset_betas = calculate_betas(df_factset_campaign_cleaned, df_factset_pricing, df_sp_pricing)
        df_factset_betas.to_csv('data/factset_campaign_betas.csv')

    if True:

        df_factset_campaign_cleaned = read_clean_table('data/clean_factset_campaign_data.csv')
        df_factset_betas = read_clean_table('data/factset_campaign_betas.csv')

        print(h1('engineering'))
        df_factset_campaign_engineered = engineer_features(df_factset_campaign_cleaned, df_factset_betas)
        write_data(df_factset_campaign_engineered, 'data/engineered_factset_campaign_data.csv')

    print(h1('complete'))

    return

if __name__ == '__main__':
    main()