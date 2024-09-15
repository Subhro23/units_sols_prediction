#!/usr/bin/env python
# coding: utf-8

# In[328]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[367]:


df = pd.read_excel('product.xlsx')
df


# In[368]:


df.info()


# In[369]:


#  New Feature Creation: Price per GB of Storage
df['Price_per_GB'] = df['Price'] / df['Storage']


# In[370]:


df


# In[371]:


plt.figure(figsize=(15,6))
sns.barplot(data=df, x='Price', y='Units_Sold')
plt.title('Price vs Units Sold')
plt.show()

1. If higher prices correspond to higher units sold, this could suggest that more expensive products are selling better.
2. If higher prices correspond to lower units sold, it indicates that as the price increases, the number of units sold    decreases.

# In[372]:


plt.figure(figsize=(15,5))
sns.barplot(data=df, x='Customer_Rating', y='Brand')
plt.title('Customer_Rating vs Brand')
plt.show()


# In[373]:


print("Missing Values:\n", df.isnull().sum())


# In[374]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,8))
sns.boxplot(data=df[['Customer_Rating', 'Battery_Life','Units_Sold','RAM','Price','Storage']])
plt.title('Box Plot of Features')
plt.show()


# In[375]:


z_scores = np.abs(stats.zscore(df[['Customer_Rating', 'Battery_Life','Units_Sold','RAM','Price','Storage']]))
df = df[(z_scores < 3).all(axis=1)] 


# In[376]:


from sklearn.preprocessing import StandardScaler

# Instantiate the StandardScaler
scaler = StandardScaler()

# Apply standardization to the specified columns
df[['RAM', 'Battery_Life', 'Price', 'Storage']] = scaler.fit_transform(df[['RAM', 'Battery_Life', 'Price', 'Storage']])


# In[377]:


# Encoding Categorical Variables
label_encoder = LabelEncoder()
df['Brand'] = label_encoder.fit_transform(df['Brand'])
df['Operating_System'] = label_encoder.fit_transform(df['Operating_System'])
df['Category'] = label_encoder.fit_transform(df['Category'])
df['Processor_Type'] = label_encoder.fit_transform(df['Processor_Type'])
df['Product_Name'] = label_encoder.fit_transform(df['Product_Name'])




# In[378]:


df


# In[379]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()


# In[380]:


from statsmodels.tools.tools import add_constant

X_vif = df[[ 'Price','Storage', 'Price_per_GB']]
X_vif_with_constant = add_constant(X_vif)

# Calculate VIF
vif = pd.DataFrame()
vif["Feature"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif_with_constant.values, i+1) for i in range(X_vif.shape[1])]

print(vif)

Price_per_GB directly incorporates the information from both Price and Storage. Keeping both Storage and Price_per_GB can lead to redundant information in This model.
Although the VIF is low, from a feature engineering perspective, it's often best to keep derived features (like Price_per_GB) and remove the original features if they convey the same information.
# In[381]:


df = df.drop(columns=['Product_ID','Launch_Year','Storage'])  # Drop  features

x = df.drop('Units_Sold', axis=1)
y = df['Units_Sold']  # Target


# In[382]:


x


# In[383]:


y


# In[384]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


# In[385]:


# Create linear regression model and fit it on training data
model = LinearRegression()
model.fit(x_train, y_train)


# In[386]:


y_test_pred = model.predict(x_test)


# In[387]:


y_test_pred


# In[388]:


y_test


# In[389]:


train_residuals = y_train - y_train_pred


# In[390]:


mse = mean_squared_error(y_test,y_test_pred)
print('MSE:',mse)

rmse = np.sqrt(mse)
print("RMSE:",rmse)

mae = mean_absolute_error(y_test,y_test_pred)
print("MAE:",mae)

r2 = r2_score(y_test,y_test_pred)
print('R2:', r2)


# In[391]:


y_train_pred = model.predict(x_train)
y_train_pred


# In[392]:


y_train


# In[393]:


mse = mean_squared_error(y_train,y_train_pred)
print('MSE:',mse)

rmse = np.sqrt(mse)
print("RMSE:",rmse)

mae = mean_absolute_error(y_train,y_train_pred)
print("MAE:",mae)

r2 = r2_score(y_train,y_train_pred)
print('R2:', r2)


# In[356]:


from sklearn.linear_model import Lasso



# Instantiate the Lasso model with a specific alpha value for L1 regularization
model = Lasso(alpha=1.0)
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print('MSE:', mse)
print("RMSE:", rmse)
print("MAE:", mae)
print('R2:', r2)


# In[394]:


from sklearn.linear_model import Ridge


# Instantiate the Ridge model with a specific alpha value for L2 regularization
model = Ridge(alpha=1.0)
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print('MSE:', mse)
print("RMSE:", rmse)
print("MAE:", mae)
print('R2:', r2)


# In[358]:


model = Ridge(alpha=1.0)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)

# Evaluate the model
mse = mean_squared_error(y_train,y_train_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train,y_train_pred)
r2 = r2_score(y_train,y_train_pred)

print('MSE:', mse)
print("RMSE:", rmse)
print("MAE:", mae)
print('R2:', r2)


# In[395]:


train_residuals = y_train - y_train_pred
train_residuals


# In[396]:


train_residuals = y_train - y_train_pred

shapiro_test = stats.shapiro(train_residuals)
print(f'Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}')

# Q-Q Plot
sm.qqplot(train_residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

The p-value is 0.001283173798583448, which is less than the commonly used significance level of 0.05 (α = 0.05).

Reject the null hypothesis; the data is not normally distributed.1. Despite the non-normal distribution of your data, the high R2 score and relatively low error metrics suggest that your Ridge regression model performs well. Continue to evaluate and improve your model based on the metrics and validation methods mentioned.

2. R2 Score: Both training and testing R2 scores are very high (close to 1), which suggests that the model explains most of the variance in the target variable. This indicates a good fit.

3. Error Metrics: Lower values of MAE and RMSE generally indicate better model performance. While the training RMSE is lower than the testing RMSE, this difference is normal and can be attributed to the model fitting better on the training data compared to the unseen testing data.
# In[397]:


import pickle
with open('ridge_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Ridge model saved to ridge_model.pkl")

# Load the Ridge model from the pickle file
with open('ridge_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Ridge model loaded from ridge_model.pkl")


# In[363]:


# df = pd.DataFrame(data)

# # Save DataFrame to CSV
# df.to_csv('product_dataa.csv', index=False)

# print("CSV file created successfully!")


# In[406]:


df1 = x
df1


# In[407]:


#df1 = pd.DataFrame(data)

# Save DataFrame to CSV
df1.to_csv('product_data3.csv', index=False)

print("CSV file created successfully!")

Generally we used linear Regreesion on normally distributed data, although The data is not normally distributed in this case, but  R2 Score: Both training and testing R2 scores are very high  which suggests that this model explains most of the variance in the target variable. This indicates a good fit.Lower values of MAE and RMSE generally indicate better model performance. While the training RMSE is lower than the testing RMSE, this difference is normal and can be attributed to the model fitting better on the training data compared to the unseen testing data.As both train and test data it is showing 1 so made a hyper parameter tuning using both Rasso and Ridge where Rasso fit well as per as r2 square,MAE,RMSE
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




