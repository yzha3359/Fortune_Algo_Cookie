# Fortune Algo Cookies

**Authors** - `Prajwal Patnaik, Bruno Ivasic, Diana Zhang, Kerry Zhang`

**Main Code** - `fortune_algo_cookies.ipynb`

**Parameter Optimisation Code** - `DMAC_RSI_Param_Optimisation.ipynb`

### Libraries and Dependencies Used

- pandas
- numpy
- matplotlib
- yfinance
- hvplot
- datetime
- holoviews
- prophet
- scikit-learn
- imbalanced-learn
- finta
- pandas-ta
- seaborn
- backtesting


## Project Overview
This project leverages the multiple skills and methodologies to analyse and develop strategies for low-risk stock using Pepsi Co. stock ('PEP') data collected from Yahoo Finance. The analysis covers a period spanning the last 5 years, with results and documentation centered around data until the 17th of December.

The approach was to start with a **Time Series Analysis**, utilising `Facebook Prophet`, that forecasts the stock price for the next 90 days. This analysis presents scenarios of best-case, worst-case, and most likely outcomes.

The next step was to implement **Algorithmic Trading Strategies**. Three different startegies were explored to understand their unique behaviours, strengths, and limitations. The strategies are namely:

* `Dual-Moving Crossover Average (DMAC)`
* `Bollinger Bands`
* `Relative Strength Index (RSI)`

Each algorithmic strategy undergoes backtesting to evaluate its risk/reward characteristics. Advantages and drawbacks of each approach are carefully analysed.

To further conceptualise the defined trading signals **Supervised Machine Learning** classifiers are implemented with the goal of determining the optimal strategy while understanding their respective performances. Again, three classifiers are used to have understand the sensitivity and specificity (precision & recall). 

* `Balanced Random Forest`
* `Support Vector Machines`
* `Ada Boost`

Additionally, efforts were made to optimise variables within the DMAC and RSI strategies, and the main code utilise the values derived from the optimisation function.


**All the results below are based on the simulation on December 17, 2023.**

## Time Series Analysis with Prophet

This step involves using the historical stock data to build a time series model to analyse and forecast patterns in the daily closing price. Here, the `Prophet` model was used to forecast the closing price for the next 90 days.

![Prophet Forecast](./Data%20Visuals/Prophet_Forecast.png)

The closing price can be predicted with 95% certainity from the prophet model.

![Prophet Forecast](./Data%20Visuals/Closing_Price_Prediction.png)

The median values for the three scenarios forecasted over the next 90 days are as follows:

* Best Case - **$173.81**
* Most Likely - **$168.12**
* Worst Case - **$162.43**


## Algorithmic Trading

### Dual-Moving Average Crossover (DMAC)

The Dual-Moving Average Crossover Trading Algorithm is a strategy that generates buy or sell signals based on the intersection of two different moving averages, typically a shorter-term average crossing above or below a longer-term average, indicating potential shifts in market trends.

The moving averages in this algorithm are taken as the exponential moving average (EMA) which is an exponentially weighted average of the previous 'n' closing periods. More recent closing prices are weighted heavier than older closing periods providing an average that is faster to respond to changing prices.

After optimisation, the short window is set to ***8*** and the long window is set to ***68***. A simple strategy is applied to generate the trade signal where `1` is generated whenever the short window moving average is greater than the long window. The entry/exit points are then created and the variables are visualised.

![DMAC](./Data%20Visuals/Entry_Exit_DMAC.png)

To backtest, an initial investment of $100,000 and a share size of 500 is assumed. Additionally, the following variables are generated. 

1. Position
2. Entry/Exit Position
3. Portfolio Holdings
4. Portfolio Cash
5. Portfolio Total
6. Portfolio Daily Returns
7. Portfolio Cumulative Returns 

The Portfolio Total is now visualised against the entry exit points to understand the actual market behaviour and compare how well the algorithm strategy performed.

![DMAC](./Data%20Visuals/Backtest_DMAC.png)

To further delineate the risk-reward characteristics of the trading algorithm the following metrics are evaluated.

| Metric    | Backtest |
| -------- | ------- |
| Annualised Returns  | 0.005192    |
| Cumulative Returns | 0.00335    |
| Annual Volatility   | 0.094973    |
| Sharpe Ratio   | 0.054665    |
| Sortino Ratio   | 0.054686   |
| Greatest Profit   | $8,614  |
| Greatest Loss  | $3,090  |

Despite being a low-risk stock, the generated annualised and cumulative returns (`0.52%` and `0.33%` respectively) are still relatively low. It suggests that while the stock may have lower volatility, the algorithm's ability to capitalise on its movements might be limited. 

The low values of both Sharpe and Sortino ratios (`0.055` approximately) suggest that when adjusting for risk, the returns achieved might not be satisfactory even in a low-risk context. This indicates the algorithm might not fully capitalise on the stability of the low-risk stock.

The annual volatility of `9.50%` indicates moderate fluctuations even in a low-risk stock. This could be due to various market factors influencing the stock's performance.

The strategy returns of the algorithm is illustrated below.

![DMAC](./Data%20Visuals/Strategy_Returns_DMAC.png)

The next step is to use machine learning to automate the trade decisions so that it can adapt to new data.

The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.

Further reading on the data split in *Appendix*.

The performance metrics of the Supervised Learning classifiers are tabulated below.

| Classifier | Balanced Accuracy | Trade Signal | Precision | Recall | F1 Score |
| ---------- | ----------------- | ------------ | --------- | ------ | -------- |
| Balanced Random Forest       | 79.12                | 0            | 72        | 66    | 69       |
|            |                   | 1            | 90        | 92     | 91       |
| Support Vector Machines     | 81.49                | 0            | 87        | 66    | 75      |
|            |                   | 1            | 91        | 97     | 94       |
| Ada Boost       | 74.52                | 0            | 69        | 56    | 62       |
|            |                   | 1            | 88        | 93     | 90       |


* SVM demonstrates the highest balanced accuracy among the three classifiers. 
* For positive Trade Signal identification (Trade Signal 1), both SVM and Balanced Random Forest display strong precision, recall, and F1 scores. 
* Ada Boost lags slightly behind the others in overall performance metrics, especially for Trade Signal 0.


### Bollinger Bands

The Bollinger Bands consist of three lines: a simple moving average (SMA) line at the centre and two additional lines above and below the SMA that represent the standard deviations (usually two) of price movements giving it a dynamic nature and versatility.

The trading signals examine the closing price against the upper and lower Bollinger Band thresholds. When the closing price falls below the lower band (BB_LOWER), it triggers a buy signal (Signal = 1) only if no existing buy signal is active (trade_signal < 1). Subsequently, it tracks the buy price.

To manage risk, it sets a stop-loss condition (`here 30%`). If the stock's price drops below a specified stop-loss percentage from the buy price, it triggers a sell signal (Signal = 0) and resets the buy signal (trade_signal = 0). Moreover, it generates a sell signal when the price rises above the upper band (BB_UPPER) after a buy signal (trade_signal > 0). The algorithm is designed to ensure only one entry and exit point per trade cycle, maintaining or closing the position based on these conditions while monitoring trade signals. 

The entry/exit points are then created and the variables are visualised.

![BB](./Data%20Visuals/Entry_Exit_BB.png)

The backtesting is performed similar to the previous algorithm (DMAC).

The Portfolio Total is visualised against the entry exit points to understand the actual market behaviour and compare how well the algorithm strategy performed.

![BB](./Data%20Visuals/Backtest_BB.png)

To further delineate the risk-reward characteristics of the trading algorithm the following metrics are evaluated.

| Metric    | Backtest |
| -------- | ------- |
| Annualised Returns  | 0.075267    |
| Cumulative Returns | 0.4105    |
| Annual Volatility   | 0.111694   |
| Sharpe Ratio   | 0.673866   |
| Sortino Ratio   | 0.673528   |
| Greatest Profit   | $7,729  |
| Greatest Loss  | $6,790  |

The provided metrics suggest a positive overall performance with modest returns (cumulative returns of `41.05%`), relatively low volatility, and acceptable risk-adjusted returns (Sharpe and Sortino Ratios, approximately `0.67`).

Given that the stock was low-risk (low beta), the strategy appears to align with this characteristic by showcasing lower volatility and reasonably positive returns.

However, whether these returns justify the risk profile of the low-risk stock depends on the investor's expectations and risk tolerance. For conservative investors seeking steady returns with lower risk exposure, the achieved metrics might align well with their objectives.

The strategy returns of the algorithm is illustrated below.

![BB](./Data%20Visuals/Strategy_Returns_BB.png)

Once again, machine learning is used to automate the trade decisions so that it can adapt to new data.

The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.

Further reading on the data split in *Appendix*.

The performance metrics of the Supervised Learning classifiers are tabulated below.

| Classifier | Balanced Accuracy | Trade Signal | Precision | Recall | F1 Score |
| ---------- | ----------------- | ------------ | --------- | ------ | -------- |
| Balanced Random Forest       | 70.31                | 0            | 84        | 73    | 78      |
|            |                   | 1            | 52        | 68     | 59       |
| Support Vector Machines     | 56.5                | 0            | 74        | 65    | 70     |
|            |                   | 1            | 37        | 48     | 42       |
| Ada Boost       | 73.17                | 0            | 85        | 79   | 82       |
|            |                   | 1            | 58        | 68     | 62       |


* Ada Boost outperforms both Balanced Random Forest and SVM in achieving a higher balanced accuracy. It also performs well with the Precision, Recall and F1 Score metrics.
* Trade Signal 0 generally shows higher Precision and Recall across all classifiers compared to Trade Signal 1. The stop loss percentage, if optimised, can result in better scores for Signal 1.


### Relative Strength Index Oscillator

The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions in the price of that security.

The RSI can do more than point to overbought and oversold securities. It can also indicate securities that may be primed for a trend reversal or corrective pullback in price. It can signal when to buy and sell. Traditionally, an RSI reading of `70` or above indicates an overbought situation. A reading of `30` or below indicates an oversold condition.

After optimisation, the upper bound, lower bound, window size are set to ***60***, ***40***, and ***14*** respectively. 

The trading signals are initialised to a neutral position and then start tracking trade signals and buy prices. The RSI thresholds (rsi_lower_bound and rsi_upper_bound) guide the buy and sell decisions: when the RSI falls below the lower threshold and no buy signal is active, it triggers a buy signal and records the buy price. Conversely, if the RSI rises above the upper threshold after a buy signal, it triggers a sell signal. 

Additionally, a stop-loss mechanism is in place: if the stock price falls below a stop-loss percentage (`here 5%`) from the buy price, it triggers a sell signal to limit potential losses. The algorithm ensures only one entry and exit point per trade cycle to effectively manage trade signals based on RSI values.

The entry/exit points are then created and the variables are visualised.

![RSI](./Data%20Visuals/Entry_Exit_RSI.png)
![RSI](./Data%20Visuals/RSI_Oscillator.png)

The backtesting is performed similar to the DMAC algorithm.

The Portfolio Total is visualised against the entry exit points to understand the actual market behaviour and compare how well the algorithm strategy performed.

![RSI](./Data%20Visuals/Backtest_RSI.png)

To further delineate the risk-reward characteristics of the trading algorithm the following metrics are evaluated.

| Metric    | Backtest |
| -------- | ------- |
| Annualised Returns  | 0.049958    |
| Cumulative Returns | 0.24795    |
| Annual Volatility   | 0.104634   |
| Sharpe Ratio   | 0.477456   |
| Sortino Ratio   | 	0.477431   |
| Greatest Profit   | $15,170  |
| Greatest Loss  | $10,514 |

The annualised return of approximately `5%` and a cumulative return of `24.8%` indicate a positive outcome over the observed period. These returns are indicative of the strategy's ability to generate profits, displaying a consistent growth in investment value.

Moreover, the annual volatility of around `10.46%` implies moderate fluctuations in returns. Given the low-risk stock, this level of volatility suggests a relatively stable performance with acceptable fluctuations within the expected range. The Sharpe and Sortino Ratios, measuring the strategy's risk-adjusted returns, stands at approximately `0.477`, positive return relative to the downside risk taken.

The strategy returns of the algorithm is illustrated below.

![RSI](./Data%20Visuals/Strategy_Returns_RSI.png)

Once again, machine learning is used to automate the trade decisions so that it can adapt to new data.

The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.

Further reading on the data split in *Appendix*.

The performance metrics of the Supervised Learning classifiers are tabulated below.

| Classifier | Balanced Accuracy | Trade Signal | Precision | Recall | F1 Score |
| ---------- | ----------------- | ------------ | --------- | ------ | -------- |
| Balanced Random Forest       | 74.73                | 0            | 91        | 83    | 87      |
|            |                   | 1            | 48        | 66     | 56       |
| Support Vector Machines     | 75.05                | 0            | 90        | 92    | 91     |
|            |                   | 1            | 63        | 58     | 61       |
| Ada Boost       | 69.4               | 0            | 88        | 92   | 90       |
|            |                   | 1            | 58        | 47     | 52       |


* Balanced Random Forest and SVM perform relatively similarly, with comparable balanced accuracy. Both show stronger precision, recall, and F1 scores for Trade Signal 0 compared to Trade Signal 1.

* Ada Boost, although slightly lower in overall balanced accuracy, demonstrates higher precision and recall for Trade Signal 0 but performs less effectively for Trade Signal 1.


## Final Notes

* Both Bollinger Bands and RSI Oscillator algorithms demonstrated good performance metrics with high cumulative returns and a positive returns relative to any downside risk(s) taken. The limitation of both these algorithms can be attributed to the false signals as observed in the ML algorithm where it indicated lower sensitivity and specificity. The Bollinger bands tend to demonstrate false signals during low volatility periods whilst the RSI remain in extreme zones for extended periods during strong trends.

* The DMAC algorithm can be further improvised with improved logics in signal generation to attribute a higher performance metric. Visibly, it did perform well in the ML implementation as it could identify the trade signals better. This can be attributed to the algorithm's clear signals where a stragihtforward approach of buy/sell signals are based on moving crossover averages.

* Whilst the DMAC and RSI strategies have additional optimisation codes to determine the window sizes, the Bollinger bands have not used any additional optimisation strategy.





## Appendix

**Using DateOffset to create rolling windows to test the model** - *The code segments the time series data into windows, moving sequentially through the dataset.*

This approach facilitates model evaluation across multiple timeframes, allowing for insights into the model's performance consistency and adaptability across different periods.

By iterating through the dataset with fixed training and testing durations, this methodology ensures an orderly assessment of the model's predictive capabilities while maintaining temporal relevance.
This rolling-window technique becomes essential in time series analysis, enabling robust evaluation and validation of predictive models within a dynamic temporal context.

**Process Overview:**

* The train_duration (3 months in this case) is set for the training period, and step_size (1 month) for the shift in window intervals.

* The code iterates through the dataset, setting the training end point (training_end) based on the current starting point.

* It generates the training data (X3_train and y3_train) for this window and appends these training dataframes to the respective lists.

* Similarly, it establishes the testing end point (testing_end) immediately following the training period.

* Creates the testing data (X3_test and y3_test) for this window and appends these testing dataframes to the respective lists.

* After the loop concludes, the code concatenates all the stored training and testing dataframes into the final training and testing sets.



