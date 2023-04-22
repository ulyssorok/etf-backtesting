# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_paths):
    """
    Read stock data from CSV files into a dictionary of DataFrames.
    
    :param file_paths: List of CSV file paths
    :return: Dictionary with stock symbols as keys and corresponding DataFrames as values
    """
    data = {}
    for file_path in file_paths:
        symbol = os.path.basename(file_path).split('.')[0]
        data[symbol] = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def calculate_adjusted_close(data):
    """
    Use the adjusted close prices for each stock in the data dictionary.
    
    :param data: Dictionary of stock DataFrames
    :return: Dictionary of stock DataFrames with adjusted close prices
    """
    for symbol, df in data.items():
        df['Adj Close'] = df['Adj Close']
    return data

def calculate_daily_returns(data):
    """
    Calculate daily returns for each stock in the data dictionary.
    
    :param data: Dictionary of stock DataFrames
    :return: Dictionary of stock DataFrames with daily returns
    """
    for symbol, df in data.items():
        df['Daily Return'] = df['Adj Close'].pct_change()
    return data

def rebalance_portfolio(data, rebalance_date):
    """
    Calculate allocations for each stock based on inverse volatility rebalancing.
    
    :param data: Dictionary of stock DataFrames
    :param rebalance_date: Date for which the portfolio is rebalanced
    :return: List of allocations for each stock
    """
    volatilities = []
    for symbol, df in data.items():
        past_month = df.loc[:rebalance_date].last('1M')
        volatilities.append(past_month['Daily Return'].std())

    inverse_volatilities = [1 / v for v in volatilities]
    total_inverse_volatility = sum(inverse_volatilities)
    allocations = [iv / total_inverse_volatility for iv in inverse_volatilities]
    
    return allocations

def backtest_portfolio(data, start_date, end_date):
    """
    Backtest the portfolio using inverse volatility rebalancing and calculate the portfolio value over time.
    
    :param data: Dictionary of stock DataFrames
    :param start_date: Start date for the backtest period
    :param end_date: End date for the backtest period
    :return: DataFrame with portfolio values over time, List of allocation history
    """
    rebalance_dates = pd.date_range(start_date, end_date, freq='BM')
    portfolio_value = pd.DataFrame(columns=['Date', 'Portfolio Value'])
    allocations_history = []

    for i, rebalance_date in enumerate(rebalance_dates):
        allocations = rebalance_portfolio(data, rebalance_date)
        allocations_history.append(allocations)

        if i == 0:
            portfolio_value = pd.concat([portfolio_value, pd.DataFrame({'Date': [rebalance_date], 'Portfolio Value': [100]})], ignore_index=True)
        else:
            prev_date = rebalance_dates[i-1]
            portfolio_value_prev = portfolio_value.loc[portfolio_value['Date'] == prev_date]['Portfolio Value'].iloc[0]
            returns = [data[symbol].loc[prev_date:rebalance_date]['Daily Return'].sum() for symbol in data.keys()]
            weighted_returns = [allocations[j] * returns[j] for j in range(len(allocations))]
            portfolio_value_new = portfolio_value_prev * (1 + sum(weighted_returns))
            portfolio_value = pd.concat([portfolio_value, pd.DataFrame({'Date': [rebalance_date], 'Portfolio Value': [portfolio_value_new]})], ignore_index=True)

    return portfolio_value, allocations_history

def calculate_sharpe_ratio(portfolio_value):
    """
    Calculate the Sharpe ratio of the portfolio.
    
    :param portfolio_value: DataFrame with portfolio values over time
    :return: Sharpe ratio
    """
    daily_returns = portfolio_value['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() * np.sqrt(252)) / daily_returns.std()
    return sharpe_ratio

def calculate_cagr(portfolio_value):
    """
    Calculate the compound annual growth rate (CAGR) of the portfolio.
    
    :param portfolio_value: DataFrame with portfolio values over time
    :return: CAGR
    """
    start_value = portfolio_value['Portfolio Value'].iloc[0]
    end_value = portfolio_value['Portfolio Value'].iloc[-1]
    years = (portfolio_value['Date'].iloc[-1] - portfolio_value['Date'].iloc[0]).days / 365
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr

def save_to_csv(data, filename):
    """
    Save DataFrame to a CSV file.
    
    :param data: DataFrame to be saved
    :param filename: Filename of the output CSV file
    """
    data.to_csv(filename, index=False)

def plot_portfolio_value(portfolio_value):
    """
    Plot portfolio value over time and save the plot as a PNG file.
    
    :param portfolio_value: DataFrame with portfolio values over time
    """
    plt.plot(portfolio_value['Date'], portfolio_value['Portfolio Value'])
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.savefig('portfolio_value.png')
    plt.show()

file_paths = ['SPY.csv', 'VGK.csv', 'EEM.csv', 'GLD.csv', 'TLT.csv']
data = read_data(file_paths)
data = calculate_adjusted_close(data)
data = calculate_daily_returns(data)
portfolio_value, allocations_history = backtest_portfolio(data, '2020-01-01', '2022-12-31')
sharpe_ratio = calculate_sharpe_ratio(portfolio_value)
cagr = calculate_cagr(portfolio_value)

print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
print(f'CAGR: {cagr:.4f}')

rebalance_dates = pd.date_range('2020-01-01', '2022-12-31', freq='BM')
allocations_df = pd.DataFrame(allocations_history, columns=data.keys(), index=rebalance_dates)
allocations_df.index.name = 'Rebalance Date'
save_to_csv(allocations_df.reset_index(), 'allocations.csv')

plot_portfolio_value(portfolio_value)