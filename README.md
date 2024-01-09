**Ads CTR Analysis stands for Click-Through Rate Analysis for advertisements. Ads CTR Analysis is the process of examining the effectiveness of online advertisements by measuring the rate at which users click on an ad’s link to reach the advertiser’s website.** <p>

# Ads CTR Forecasting: Process We Can Follow <p>
**Ads CTR Analysis and Forecasting are crucial for businesses to assess the return on investment (ROI) of their advertising efforts and make data-driven decisions to improve ad performance. Below are the steps we can follow for the task of Ads CTR Analysis and Forecasting:** <p>

1. **Gather ad data, including the number of ad impressions (how often an ad was shown), the number of clicks, and any other relevant metrics.** <p>
2. **Explore the data to understand its characteristics and distribution. Calculate basic statistics, such as the mean CTR (Click-Through Rate) and standard deviation.** <p>
3. **Create visualizations, such as line charts or bar graphs, to represent CTR trends over time.** <p>
4. **Conduct A/B tests if necessary to compare the performance of different ad variations.** <p>
5. **Analyze the CTR data to identify factors that influence ad performance.** <p>
6. **Build a forecasting model to predict future CTR values.** 

# Ads CTR Forecasting using Python <p>
**Let’s get started with the task of Ads CTR Analysis and forecasting by importing the necessary Python libraries and the dataset:** <p>

import pandas as pd <p> 
import matplotlib.pyplot as plt <p>
import seaborn as sns <p>
import numpy as np <p>
import plotly.graph_objects as go <p>
import plotly.express as px <p>

**Reading of Dataset** <p>
df1 = pd.read_csv("ctr.csv") <p>
**Head of the dataset** <p>
df1.head() <p>
![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/5b3d68f8-b783-4a23-8f1c-81b99dda6c88)

**Tail of the Dataset** <p>
df1.tail() <p>
![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/1b899596-ec40-421a-a406-8f8e6478db6a)

**Descriptive statistics of dataset** <p>
df1.info() <p>
![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/3685f303-3b55-4f3d-b0eb-841de10ee5f9)

df1.describe() <p>
![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/2eccb345-85ac-436c-8c22-45e9cf5a86e2)

**Now, let’s visualize the clicks and impressions over time:** <p>

fig = go.Figure() <p>
fig.add_trace(go.Scatter(x=df1.index, y=df1['Clicks'], mode='lines', name='Clicks')) <p>
fig.add_trace(go.Scatter(x=df1.index, y=df1['Impressions'], mode='lines', name='Impressions')) <p>
fig.update_layout(title='Clicks and Impressions Over Time') <p>
fig.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/08a5d7e5-7016-415c-99f4-3e5df0dc51ff)

**Now, let’s have a look at the relationship between clicks and impressions:** <p>

sns.pairplot(df1) <p>
plt.title('Relationship between Clicks and Impressions') <p>
plt.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/2a98e2f8-9176-476a-a48c-16cff8d84738)

**So, the relationship between clicks and impressions is linear. It means higher ad impressions result in higher ad clicks. Now, let’s calculate and visualize CTR over time:** <p>

df1['CTR'] = (df1['Clicks'] / df1['Impressions']) * 100 <p>
fig = px.line(df1, x=df1.index, y='CTR', title='Click-Through Rate (CTR) Over Time') <p>
fig.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/de5079ba-3b9a-4d0c-a15d-224441fcecf5)


df1.index = pd.to_datetime(df1.index) <p>

**Create new columns** <p>
df1['DayOfWeek'] = df1.index.dayofweek <p>
df1['WeekOfMonth'] = df1.index.to_series().dt.isocalendar().week // 4  # Use isocalendar().week <p>

**Create a new column 'DayCategory' to categorize weekdays and weekends** <p>
df1['DayCategory'] = df1['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday') <p>

**Create a bar plot to compare CTR on weekdays vs. weekends** <p>
fig = px.bar(ctr_by_day_category, x='DayCategory', y='CTR', title='CTR on Weekdays', <p>
             labels={'CTR': 'Average CTR'}) <p>

**Customize the layout** <p>
fig.update_layout(yaxis_title='Average CTR') <p>

**Show the plot** <p>
fig.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/0935ddf4-7ddf-4975-b649-4c5c1a66d5c6)

**Now, let’s compare the CTR on weekdays and weekends:** <p>

**Group the data by 'DayCategory' and calculate the sum of Clicks and Impressions for each category**<p>
grouped_df1 = df1.groupby('DayCategory')[['Clicks', 'Impressions']].sum().reset_index() <p>

**Create a grouped bar chart to visualize Clicks and Impressions on weekdays** <p>
fig = px.bar(grouped_df1, x='DayCategory', y=['Clicks', 'Impressions'], <p>
             title='Impressions and Clicks on Weekdays', <p>
             labels={'value': 'Count', 'variable': 'Metric'}, <p>
             color_discrete_sequence=['blue', 'green']) <p>

**Customize the layout** <p>
fig.update_layout(yaxis_title='Count') <p>
fig.update_xaxes(title_text='Day Category') <p>

**Show the plot** <p>
fig.show() <p>

# Ads CTR Forecasting <p>
**Now, let’s see how to forecast the Ads CTR. As CTR is dependent on impressions and impressions change over time, we can use Time Series forecasting techniques to forecast CTR. As CTR is seasonal, let’s calculate the p, d, and q values for the SARIMA model:**<p>
df1.reset_index(inplace=True) <p>

from statsmodels.tsa.arima.model import ARIMA <p>
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf <p>

**resetting index** <p>
time_series = df1.set_index('Date')['CTR'] <p>

**Differencing** <p>
differenced_series = time_series.diff().dropna() <p>

**Plot ACF and PACF of differenced time series** <p>
fig, axes = plt.subplots(1, 2, figsize=(12, 4)) <p>
plot_acf(differenced_series, ax=axes[0]) <p>
plot_pacf(differenced_series, ax=axes[1]) <p>
plt.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/696f5fbf-56ed-4319-8ce1-b697b04dbd92)

**The value of p, d, and q will be one here. You can learn more about calculating p, d, and q values from here. And as we are using the SARIMA model here, the value of s will be 12.** <p>

**Now, let’s train the forecasting model using SARIMA:** <p>

from statsmodels.tsa.statespace.sarimax import SARIMAX <p>

p, d, q, s = 1, 1, 1, 12 <p>

model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(p, d, q, s)) <p>
results = model.fit() <p>
print(results.summary()) <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/7f40a333-b9fa-47dd-bd8e-fdf96da784d2)

**Now, here’s how to predict the future CTR values:** <p>

future_steps = 100 <p>
predictions = results.predict(len(time_series), len(time_series) + future_steps - 1) <p>
print(predictions) <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/4f37b985-afe7-49b3-8d68-1da91d58325e)

**Now, let’s visualize the forecasted trend of CTR:** <p>

forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions}) <p>

**Plot the original data and predictions** <p>
fig = go.Figure() <p>

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Predictions'], <p>
                         mode='lines', name='Predictions')) <p>

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Original'], <p>
                         mode='lines', name='Original Data')) <p>

fig.update_layout(title='CTR Forecasting', <p>
                  xaxis_title='Time Period', <p>
                  yaxis_title='Impressions', <p>
                  legend=dict(x=0.1, y=0.9), <p>
                  showlegend=True) <p>

fig.show() <p>

![image](https://github.com/KalyanKumarBhogi/Advertisements_CTR-Click_time_review-_Analysis_and_Forecasting/assets/144279085/f0bfdd9b-0e7d-4174-80b2-29dc8a64a0b8)


# Conclusion <p>
**Ads CTR Analysis and Forecasting using Python offer a comprehensive approach to assess the effectiveness of online advertisements and predict future Click-Through Rates (CTR). The process involves gathering ad data, exploring its characteristics, and visualizing trends over time. Through A/B testing and analysis, factors influencing ad performance are identified. The article demonstrates the application of Time Series forecasting techniques, specifically SARIMA modeling, to predict future CTR values. Visualizations and insights, such as the linear relationship between clicks and impressions, average CTR by day of the week, and comparisons between weekdays and weekends, provide valuable information for businesses to make data-driven decisions in optimizing their advertising strategies and improving return on investment.** <p>
