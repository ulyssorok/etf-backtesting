# Inverse Volatility Portfolio Backtesting

This Python script backtests a portfolio of five ETFs using the Inverse Volatility method for monthly rebalancing. The script calculates the Sharpe Ratio and Compound Annual Growth Rate (CAGR) of the portfolio for the specified period (2020-01-01 to 2022-12-31).

The portfolio consists of the following ETFs:
- SPY
- VGK
- EEM
- GLD
- TLT

Daily OHLC data for these ETFs are provided as CSV files.

## Features

The script performs the following steps:

1. Reads stock data from CSV files.
2. Calculates adjusted close prices.
3. Calculates daily returns.
4. Rebalances the portfolio using the Inverse Volatility method.
5. Backtests the portfolio and calculates the portfolio value over time.
6. Calculates the Sharpe Ratio and CAGR of the portfolio.
7. Saves the rebalancing dates and allocations to a CSV file.
8. Plots the portfolio value over time.

## Usage

To use the script, make sure that the CSV files for the ETFs are in the correct format and available in the `data` folder. Also, ensure that you have the necessary Python packages installed (`pandas`, `numpy`, and `matplotlib`).

Run the script, and it will generate the following outputs:

1. A plot of the portfolio value over time (saved in the `output` folder as `portfolio_value.png`).
2. A CSV file with the rebalancing dates and corresponding allocations (saved in the `output` folder as `allocations.csv`).

The script will also print the Sharpe Ratio and CAGR values to the console.

## Dependencies

- pandas
- numpy
- matplotlib

## Installation

1. Clone the repository
2. Install the required packages
