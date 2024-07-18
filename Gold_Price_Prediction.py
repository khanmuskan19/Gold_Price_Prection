#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from forex_python.converter import CurrencyRates
from datetime import datetime
import plotly_express as px


# In[2]:


# !pip install forex-python


# In[3]:


data=pd.read_csv(r"C:\Users\Muskan Khan\OneDrive\Documents\JUPYTER\Gold Price Prediction\gld_price_data.csv")
print(data)


# In[4]:


data.head()


# In[5]:


# # coverting date formate from dd/mm/yyyy to mm/dd/yyyy(cz forex_python works on this format)


# def reformat_date(date_str):
#     try:
#         # Parse date string as dd/mm/yyyy format
#         dt=datetime.strptime(date_str, '%d/%m/%y')
        
#          # Format date as mm/dd/yyyy
#         return dt.strftime('%m/%d/%y')
#     except ValueError as e:
#         print(f"Error converting date {date_str}: {str(e)}")
#         return None
                    
# data['Date']=data['Date'].apply(reformat_date)
# data.head()


# In[6]:


c= CurrencyRates()

def usd_to_inr(date):
    try:
        return c.get_rate('USD','INR',date)
    except:
        return None
    
#  Creating function to fetch the exchange rates and converting to INR from USD

data['USD/INR']=data['Date'].apply(usd_to_inr)*data['EUR/USD']



# # For Memory
# 
# . EUR/USD means we've converted the EUR to USD. So it's USD.
# 
# . 1 USD value=INR/USD (60/1=1)
# 
# . So from the we're fetching the INR/USD i.e. Indian currency rates respectively by the date.
# 
# . Indian Rs.=USD/INR (1/60 Rs.)
# 
# . And we're converting for that we're just fetching the INR value at the given date.
# 

# In[7]:


data.head()
data.tail()


# In[8]:


fig=px.scatter(data, x='GLD', y='EUR/USD',color='SLV',size='USO',trendline='ols',  labels={'GLD': 'Gold Price (USD/oz)', 'EUR/USD': 'EUR/USD Exchange Rate', 'SLV': 'Silver Price (USD/oz)', 'USO': 'Oil Price (USD/barrel)'})
fig.show()


# Adding Color Scale 

# In[9]:


fig=px.scatter(data, x='EUR/USD', y='GLD',color='SLV',size='USO',trendline='ols', 
               labels={'GLD': 'Gold Price (USD/oz)', 'EUR/USD': 'EUR/USD Exchange Rate', 'SLV': 'Silver Price (USD/oz)', 'USO': 'Oil Price (USD/barrel)'}, color_continuous_scale='Viridis')
fig.show()


# In[10]:


fig=px.scatter(data, x='EUR/USD', y='USO',color='SLV',size='GLD',trendline='ols',
              labels={'GLD': 'Gold Price (USD/oz)', 'EUR/USD': 'EUR/USD Exchange Rate', 'SLV': 'Silver Price (USD/oz)', 'USO': 'Oil Price (USD/barrel)'}, 
               color_continuous_scale='Cividis', title= 'Relationship Between Gold Price, EUR/USD Exchange Rate, Silver Price, and Oil Price' )
fig.show()


# In[11]:


fig=px.scatter(data, x='GLD', y='USO',color='SLV',size='GLD',trendline='ols',
              labels={'GLD': 'Gold Price (USD/oz)', 'EUR/USD': 'EUR/USD Exchange Rate', 'SLV': 'Silver Price (USD/oz)', 'USO': 'Oil Price (USD/barrel)'}, 
               color_continuous_scale='Cividis', title= 'Relationship Between Gold Price, EUR/USD Exchange Rate, Silver Price, and Oil Price' )
fig.show()


# In[12]:



fig=px.histogram(data, x='GLD',title='Gold Price')
fig.show()


# In[13]:


data['Date'] = pd.to_datetime(data['Date'])
data['Year']=data['Date'].dt.year
data.head()


# In[14]:


# Group the data by year and calculate the mean gold price for each year
mean_gold_price_per_year=data.groupby('Year')['GLD'].max().reset_index()

fig=px.bar(mean_gold_price_per_year, y='GLD',x='Year',
                labels={'GLD':'Max Gold Price','Year':'Year'}, title='Max Gold Price per Year')

# Adjusting x-axis properties
fig.update_layout(
    xaxis=dict(
        tickmode='linear',  # Set tick mode to linear to show all ticks
        dtick=1,            # Set the distance between ticks
        tickangle=45        # Rotate x-axis labels for better readability
    )
)

fig.show()


# In[15]:


correlation= data.corr()
print(correlation['GLD'].sort_values(ascending=False))


# In[16]:


data.head()


# # Train-Test Splitting

# In[17]:


from sklearn.model_selection import train_test_split
x= np.array(data[['SPX','USO','SLV','EUR/USD']])
y=np.array(data['GLD'])
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.10)


# # Creating Pipeline
# 

# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[19]:


pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    
])

x_train_new=pipeline.fit_transform(x_train)


# # Price Prediction

# In[20]:


# selecting a desired model


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(x_train_new,y_train)



# In[21]:


#Select some test data for predictions using array slicing(test data lekr testing kr rhe instead of inputting mannually)

some_x_data=x_test[:5]
some_y_data=y_test[:5] #x and y are NumPy arrays, and NumPy arrays do not have the iloc attribute so we're using slicing. So we'll create pipeline
prepared_data=pipeline.transform(some_x_data) # if u don't wanna create pipeline here, don't!


# # Predictions

# In[22]:


# model.predict(some_x_data) test ki predictions hn
predictions=model.predict(some_x_data)
print(predictions) #this is our prediction value
list(some_y_data) #and this is our actual value of those(x_train new)


# In[23]:


comparison=pd.DataFrame({'Actual':some_y_data,'Predicted':predictions})
print(comparison)


# # Measuring the model's Accuracy

# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
my_pred_model=model.predict(x_train_new)
mae=mean_absolute_error(y_train, my_pred_model)
mse=mean_squared_error(y_train,my_pred_model)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)


# In[25]:


rmse=np.sqrt(mse)
print(rmse)


# # Testig the model on test data

# In[26]:


x_test_preapred=pipeline.transform(x_test)
test_predictions=model.predict(x_test_preapred)

comparison=pd.DataFrame({'Actual':y_test,'Predicted':test_predictions})
test_mae=mean_absolute_error(y_test,test_predictions)
test_mse=mean_squared_error(y_test,test_predictions)
test_rmse=np.sqrt(test_mse) 

print(comparison)


# In[27]:


print(test_mse)


# In[28]:


print(test_rmse)


# In[29]:


print(test_mae)


# # Cross Validation 

# In[30]:


from sklearn.model_selection import cross_val_score
cross_scores=cross_val_score(model,x_train_new,y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores=np.sqrt(-cross_scores)


# In[31]:


print(rmse_scores)


# In[32]:


y_train.min()
# y_train.max()


# In[33]:


y_train.max()


# In[34]:


# Range of the target variable is from 70 to 184. So the rmse is excellent.


# # Saving it for the  Future

# In[35]:


from joblib import dump,load
dump(model,'Gold_model.joblib')


# In[36]:


# data.head()


# # Testing of Joblib Saved Model

# In[37]:


model=load("Gold_model.joblib")
features=np.array([[1007.160034, 64.860001,78.470001,15.180,]])
model.predict(features)


# In[ ]:





# In[ ]:




