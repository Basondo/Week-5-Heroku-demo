import numpy as np
import pandas as pd
import pickle

st = pd.read_csv('Stores copy.csv')
stores = st.rename(columns = {'Daily_Customer_Count': 'Daily_Customers', 
                              'Store_Sales': 'Sales'})

X = stores.iloc[:, 2:4]
y = stores.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('store_lst.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('store_lst.pkl','rb'))
print((model.predict([[10, 1]])))