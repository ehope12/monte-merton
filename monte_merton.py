import pandas as pd
from my_functions import optimize_and_plot, show_epsilon, interactive_stock_plot

"""
The script simulates a Merton Jump Diffusion (MDJ) process for a given stock ticker.

The first box sets initial sample parameters.
    The 'ticker' variable sets the stock ticker.
    The 'start' variable sets the start date for the data retrieval.
    The 'end' variable sets the end date for the data retrieval. The end graph will begin where the data retrieval ends and end at the current day.
    The 'sims' variable sets the number of simulations to run.
The second box creates a representation of the epislon value, which determines the threshold for a jump.
The third box shows the MJD process for the given stock ticker with optimized parameters.
The fourth box shows a more interactive plot with the MJD process.
The fifth box shows a comparison between a simulated random walk and a prediction from an off-the-shelf ML model.
"""


if __name__ == "__main__":
    """ BOX 1 """
    # -----------------------------------------------------------------------------------------------------------------------------------#
    ticker = 'SPY'
    start = '2000-01-01'
    end = (pd.to_datetime('today') - pd.DateOffset(years=1)).strftime('%Y-%m-%d')  # Exclude the last year (or whatever time interval) for comparison
    sims = 20
    # -----------------------------------------------------------------------------------------------------------------------------------#

    """ BOX 2 """
    # -----------------------------------------------------------------------------------------------------------------------------------#
    # show_epsilon(ticker, start, end)
    # -----------------------------------------------------------------------------------------------------------------------------------#

    """ BOX 3 """
    # -----------------------------------------------------------------------------------------------------------------------------------#
    # optimize_and_plot(ticker, start, end, sims)
    # -----------------------------------------------------------------------------------------------------------------------------------#

    """ BOX 4 """
    # -----------------------------------------------------------------------------------------------------------------------------------#
    interactive_stock_plot(ticker, start, end)
    # -----------------------------------------------------------------------------------------------------------------------------------#