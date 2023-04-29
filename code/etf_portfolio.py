import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'output'


def data_file_path(filename):
    """
    Create a complete file path for the given filename in the data folder.

    :param filename: The name of the data file
    :return: Full path to the data file
    """
    return os.path.join(DATA_FOLDER, filename)


def output_file_path(filename):
    """
    Create a complete file path for the given filename in the output folder.

    :param filename: The name of the output file
    :return: Full path to the output file
    """
    return os.path.join(OUTPUT_FOLDER, filename)


def read_data(file_paths):
    """
    Read adjusted close price data from CSV files into a DataFrame.

    :param file_paths: List of CSV file paths
    :return: DataFrame with dates as index and stock symbols as columns
    """
    data = {
        os.path.basename(file_path).split('.')[0]: pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')['Adj Close']
        for file_path in file_paths
    }
    return pd.DataFrame(data)


def calculate_daily_returns(data):
    """
    Calculate daily returns for each stock in the DataFrame.

    :param data: DataFrame with stock price data
    :return: DataFrame with daily returns for each stock
    """
    return data.pct_change()


def rebalance_portfolio(data, rebalance_date):
    """
    Calculate portfolio allocations based on the inverse volatility of each stock.

    :param data: DataFrame with daily returns for each stock
    :param rebalance_date: Date to rebalance the portfolio
    :return: Series with portfolio allocations for each stock
    """
    past_month = data.loc[:rebalance_date].last('1M').iloc[:-1]
    volatilities = past_month.std()
    inverse_volatilities = 1 / volatilities
    total_inverse_volatility = inverse_volatilities.sum()
    return inverse_volatilities / total_inverse_volatility


def backtest_portfolio(data, start_date, end_date):
    """
    Backtest the portfolio using inverse volatility-based allocations.

    :param data: DataFrame with daily returns for each stock
    :param start_date: Start date of the backtest
    :param end_date: End date of the backtest
    :return: DataFrame with portfolio value over time and a list of allocation history
    """
    rebalance_dates = pd.date_range(start_date, end_date, freq='B')
    portfolio_value = pd.DataFrame(columns=['Date', 'Portfolio Value'])
    allocations_history = []

    for i, rebalance_date in enumerate(rebalance_dates):
        allocations = rebalance_portfolio(data, rebalance_date)
        allocations_history.append(allocations)

        if i == 0:
            current_portfolio_value = 100
        else:
            prev_date = rebalance_dates[i - 1]
            prev_portfolio_value = portfolio_value.loc[portfolio_value['Date'] == prev_date, 'Portfolio Value'].iloc[0]
            returns = data.loc[prev_date:rebalance_date].apply(lambda x: np.prod(1 + x) - 1)
            current_portfolio_value = prev_portfolio_value * (1 + (allocations * returns).sum())

        new_entry = pd.DataFrame({'Date': [rebalance_date], 'Portfolio Value': [current_portfolio_value]})
        portfolio_value = pd.concat([portfolio_value, new_entry], ignore_index=True)

    return portfolio_value, allocations_history


def calculate_sharpe_ratio(portfolio_value):
    """
    Calculate the Sharpe ratio for the given portfolio value.

    :param portfolio    value: DataFrame with portfolio value over time
    :return: Sharpe ratio of the portfolio
    """
    daily_returns = portfolio_value['Portfolio Value'].pct_change().dropna()
    return (daily_returns.mean() * np.sqrt(252)) / daily_returns.std()


def calculate_cagr(portfolio_value):
    """
    Calculate the compound annual growth rate (CAGR) for the given portfolio value.

    :param portfolio_value: DataFrame with portfolio value over time
    :return: CAGR of the portfolio
    """
    start_value, end_value = portfolio_value['Portfolio Value'].iloc[[0, -1]]
    years = (portfolio_value['Date'].iloc[-1] - portfolio_value['Date'].iloc[0]).days / 365
    return (end_value / start_value) ** (1 / years) - 1


def save_to_csv(data, filename):
    """
    Save the given data to a CSV file.

    :param data: DataFrame to save as a CSV file
    :param filename: Name of the output file
    """
    data.to_csv(filename, index=False)


def plot_portfolio_value(portfolio_value):
    """
    Plot the portfolio value over time and save the plot as an image file.

    :param portfolio_value: DataFrame with portfolio value over time
    """
    plt.plot(portfolio_value['Date'], portfolio_value['Portfolio Value'])
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.savefig(output_file_path('portfolio_value.png'))
    plt.show()


def main():
    """
    Main function to run the script.
    """
    file_paths = [data_file_path(f) for f in ['SPY.csv', 'VGK.csv', 'EEM.csv', 'GLD.csv', 'TLT.csv']]
    data = read_data(file_paths)
    data = calculate_daily_returns(data)
    portfolio_value, allocations_history = backtest_portfolio(data, '2020-01-01', '2022-12-31')
    sharpe_ratio = calculate_sharpe_ratio(portfolio_value)
    cagr = calculate_cagr(portfolio_value)

    print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
    print(f'CAGR: {cagr:.4f}')

    rebalance_dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')
    allocations_df = pd.DataFrame(allocations_history, columns=data.keys(), index=rebalance_dates)
    allocations_df.index.name = 'Rebalance Date'
    save_to_csv(allocations_df.reset_index(), output_file_path('allocations.csv'))

    plot_portfolio_value(portfolio_value)


if __name__ == '__main__':
    main()

