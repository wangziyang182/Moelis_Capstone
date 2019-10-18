import logging as logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


dict_manual_renamings = {
    'campaign_symbol': 'campaign_id',
    'activist': 'activist_name',
    'campaign_announce_date': 'campaign_announcement_date',
    'in_force_prior_to_announcement_poison_pill': 'poison_pill_in_force_prior_to_announcement',
    'adopted_in_response_to_campaign_poison_pill': 'poison_pill_adopted_in_response_to_campaign',
}

list_drop_columns = [
    
    # repeat of campaign_announce_date
    'announcement_date_date', 
    
    # unused for now
    '18_months_pre_date_stock_price',
    '1_year_pre_date_stock_price',
    '6_months_pre_date_stock_price',
    '90_days_pre_date_stock_price',
    
    '18_months_pre_date_dividends',
    '1_year_pre_date_dividends',
    '6_months_pre_date_dividends',
    '90_days_pre_date_dividends',
    
    '6_months_post_date_stock_price',
    '1_year_post_date_stock_price',
    '18_months_post_date_stock_price',
    
    '6_months_post_date_dividends',
    '1_year_post_date_dividends',
    '18_months_post_date_dividends'
    
]

list_percentage_columns = [
    '18_months_pre_date_total_return',
    '1_year_pre_date_total_return',
    '6_months_pre_date_total_return',
    '90_days_pre_date_total_return',
    'ownership_pecent_on_announcements',
    '6_months_post_date_total_return',
    '1_year_post_date_total_return',
    '18_months_post_date_total_return',
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
    'ownership_pecent_on_announcements',
    
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
    '18_months_pre_announcement_date',
    '1_year_pre_announcement_date',
    '6_months_pre_announcement_date',
    '90_days_pre_announcement_date',
    '18_months_pre_date_total_return',
    '1_year_pre_date_total_return',
    '6_months_pre_date_total_return',
    '90_days_pre_date_total_return',

    # campaign success outcomes
    'number_of_board_seats_gained', 
    'proxy_campaign_winner_or_result',
    'activist_campaign_results',

    # return success outcomes
    '6_months_post_announcement_date',
    '1_year_post_announcement_date',
    '18_months_post_announcement_date',    
    '6_months_post_date_total_return',  
    '1_year_post_date_total_return', 
    '18_months_post_date_total_return' 

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
    )


def read_factset_campaign_data():
    df_raw = pd.read_excel('data/FactSet_Campaign v8.xlsx', skiprows=2, na_values=['', ' ', '_', '-', 'na', 'NA', 'n.a.'])
    return df_raw

def clean_data(df_raw):

    df_cleaning = (
        df_raw
        # rename 
        .rename(columns=clean_column_name)
        .rename(columns=dict_manual_renamings)
        # drop
        .drop(axis='columns', labels=list_drop_columns)
        # convert strings to dates based on format
        # note this fails silently for malformed dates for now
        .assign(campaign_announcement_date=lambda df: pd.to_datetime(df.campaign_announcement_date, format='%Y%m%d'))
        .assign(first_trade_date=lambda df: pd.to_datetime(df.first_trade_date, format='%Y-%m-%d %H:%M:%S'))
        .assign(last_trade_date=lambda df: pd.to_datetime(df.last_trade_date.astype(str), format='%m/%d/%Y', errors='coerce'))
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

    # from percentages to raw units
    for percentage_column in list_percentage_columns:
        df_cleaning[percentage_column] = df_cleaning[percentage_column] / 100

    # reorder
    df_cleaning = (
        df_cleaning
        .reindex(columns=list_column_order)
        .sort_values(['campaign_id', 'campaign_announcement_date', 'campaign_title'])
    )
        
    df_clean = df_cleaning

    return df_clean

def engineer_features(df_clean):

    # fundamentals
    df_engineered = (
        df_clean
        .assign(earnings_yield_at_announcement=lambda df: df.ltm_eps_at_announcement / df.price_at_announcement)
    )

    # campaign tactics
    # originally provided as a single column with each value being a tuple of tactics
    # explode this tuple into columns, then pivot so each tactic gets one column
    # then create dummies for whether a tactic was used by an activist for a given campaign
    # merge back onto the main data set
    df_tactic = (
        df_clean
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
    df_engineered = pd.merge(
        df_engineered, df_tactics_indicators,
        on='campaign_id', how='left'
    )

    # past successes
    # create a simple measure of success like whether one year future returns were positive
    # count the cumulative successes by activist
    # make sure to lag our knowledge of successes by more than one year to ensure no look-ahead bias
    # merge back onto the main data set
    df_successes = (
        df_clean
        .assign(is_return_success=lambda df: 1 * (df['1_year_post_date_total_return'] > 0))
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
    df_engineered = pd.merge_asof(
        df_engineered.sort_values('campaign_announcement_date'),
        df_cumulative_successes[['activist_id', 'lagged_campaign_announcement_date', 'past_return_successes']].sort_values('lagged_campaign_announcement_date'),
        by=['activist_id'], left_on=['campaign_announcement_date'], right_on=['lagged_campaign_announcement_date']
    )

    return df_engineered

def write_data(df, write_path):
    df.to_csv(write_path, index=False)
    return

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )
    logger.info('reading data')
    df_raw = read_factset_campaign_data()
    logger.info('cleaning data')
    df_clean = clean_data(df_raw)
    write_data(df_clean, 'data/clean_factset_campaign_data.csv')
    logger.info('engineering features')
    df_engineered = engineer_features(df_clean)
    write_data(df_engineered, 'data/engineered_factset_campaign_data.csv')
    return

if __name__ == '__main__':
    main()