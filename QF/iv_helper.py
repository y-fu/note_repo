import yfinance as yf
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

DATE_FORMATE = '%Y-%m-%d'

def get_asset(ticker):
    return yf.Ticker(ticker)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def get_atm_estimator(asset, expiry_date):
    ed_string = expiry_date.strftime(DATE_FORMATE)
    options = asset.option_chain(ed_string)
    calls = options.calls
    puts = options.puts

    # get all striks and associated implied vol for call options and put options
    strikes = np.concatenate([calls['strike'], puts['strike']])
    ivs = np.concatenate([calls['impliedVolatility'].fillna(0), puts['impliedVolatility'].fillna(0)])

    # Sort by strike price
    sorted_indices = np.argsort(strikes)
    strikes = strikes[sorted_indices]
    ivs = ivs[sorted_indices]

    params, _ = curve_fit(quadratic, strikes, ivs)
    return params

def get_asset_price_for_given_date(asset, query_date):
    # Assumption the query_date is within 1 month time.
    # Close price is used as asset price

    retrieve_asset_data = asset.history(period='1mo').reset_index()
    retrieve_asset_data['Date'] = retrieve_asset_data['Date'].apply(lambda dt: dt.date())
    retrieve_asset_data = retrieve_asset_data[retrieve_asset_data['Date'] == query_date]
    retrieve_asset_price = retrieve_asset_data.iloc[-1]['Close']
    return retrieve_asset_price

def get_implied_vol(asset, asset_price, target_date):
    
    expiry_dates = asset.options
    expiry_dates = [datetime.strptime(ed, DATE_FORMATE).date() for ed in expiry_dates]

    if target_date in expiry_dates:
        atm_estimator_params = get_atm_estimator(asset, target_date)
        atm_iv = quadratic(asset_price, *atm_estimator_params)
    else:
        # get options with expiry date just befor and after target date
        # get atm iv for the before and after expiry date
        # interpolate with before and after iv
        before = [ed for ed in expiry_dates if ed < target_date][-1]
        after = [ed for ed in expiry_dates if ed > target_date][0]

        before_estimator_params = get_atm_estimator(asset, before)
        after_estimator_params = get_atm_estimator(asset, after)
        
        before_atm_iv = quadratic(asset_price, *before_estimator_params)
        after_atm_iv = quadratic(asset_price, *after_estimator_params)
        atm_iv = (target_date - before).days/(after - before).days * (after_atm_iv - before_atm_iv) + before_atm_iv

    return atm_iv 