# monte-merton

A script for simulating a Merton Jump Diffusion (MDJ) process for a given stock ticker. The script uses data from Yahoo Finance to estimate the parameters of the MDJ model, then uses that model to plot price predictions.

## Getting Started

### Dependencies

* Python
* Numpy
* Scipy
* Matplotlib
* Math
* Yfinance
* Pandas
* Seaborn

### Usage

Once you have all of the listed dependencies, clone the repo. Then run the Python script:
```
python monte_merton.py
```

(It may take a minute to run due to the initial parameter estimations). Change the ticker in Box 1 in "monte_merton.py" to get data from a different company/ETF. Use the interactive visualization to see how changing the parameters of the MDJ model impacts the simulated prices. Drag the slider at the bottom of the screen to change the number of simulations.

## Authors

Emily Hsieh
