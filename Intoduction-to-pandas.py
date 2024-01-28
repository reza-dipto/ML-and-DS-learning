#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # There are 2 main data types in Pandas
#  1 . Series ( 1 dimensional)
#  
#  2 . dataframe ( 2 dimensional)

# # Series data type

# In[2]:


wwe_wrestlers=pd.Series(["Austin","Cena","Rock"])


# In[3]:


wwe_wrestlers


# In[4]:


entrance_theme=pd.Series(["Glass Shatter","My Time is Now","Electrifying"])
entrance_theme


# # Data frame data type 

# In[5]:


theme_song= pd.DataFrame({"name": wwe_wrestlers,"themes": entrance_theme})
theme_song


# # Import data 

# In[6]:


car_sales=pd.read_csv("007 car-sales.csv")


# In[7]:


car_sales


# # Export a  dataframe

# In[8]:


car_sales.to_csv("new-exported-car-sales.csv",index=False)


# In[9]:


new_exported_car_sales=pd.read_csv("new-exported-car-sales.csv")
new_exported_car_sales


# # here we can see an extra column "unnamed". to remove this,we add index=False while exportingbbB

# ## Describe Data

# In[10]:


new_exported_car_sales.dtypes


# In[11]:


car_sales.columns


# In[12]:


data_columns=car_sales.columns


# In[13]:


data_columns


# In[14]:


car_sales.describe()


# In[15]:


car_sales.info()


# In[16]:


car_sales.sum()


# In[17]:


car_sales["Odometer (KM)"].sum()


# # Viewing & Selecting Data

# In[18]:


car_sales.head()


# In[19]:


car_sales.tail()


# ## .loc and .iloc

# In[20]:


friends=pd.Series(["Tanmoy","Abeer","Naimul","Rigan","Mridul"],
                  index=[0,3,11,8,7])
friends


# In[21]:


friends.loc[3]


# In[22]:


car_sales.loc[4]


# In[23]:


friends.iloc[2]


# In[24]:


friends


# In[25]:


friends.loc[:3]


# In[26]:


friends.iloc[:3]


# In[27]:


car_sales


# In[28]:


car_sales.Make


# In[29]:


car_sales["Colour"]


# In[30]:


car_sales[car_sales["Make"]=="Honda"]


# In[31]:


car_sales[car_sales["Make"]=="Toyota"]


# In[32]:


car_sales[car_sales["Doors"]>=3]


# In[33]:


car_sales[car_sales["Price"]=="$4,000.00"]


# In[34]:


car_sales


# In[35]:


pd.crosstab(car_sales["Make"],car_sales["Price"])


# In[36]:


car_sales.groupby(["Make"]).max()


# In[37]:


car_sales["Odometer (KM)"].hist()


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[39]:


car_sales["Odometer (KM)"].plot()


# In[40]:


// this doesnt work to convert object column to int
car_sales["Price"]=car_sales["Price"].str.replace('[\$\,\.]','').astype(int)


# # This works to convert object column to int

# In[ ]:


car_sales["Price"].astype(str).astype(int)


# ## Manipulating Data

# In[ ]:


car_sales


# In[ ]:


car_sales["Make"]=car_sales["Make"].str.lower()


# In[ ]:


car_sales


# In[ ]:


car_sales["Make"]=car_sales["Make"].str.upper()


# In[41]:


car_sales


# In[45]:


car_sales_missing=pd.read_csv("009 car-sales-missing-data.csv")
car_sales_missing


# In[46]:


car_sales_missing["Odometer"].mean()


# In[51]:


car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean(),inplace=True)


# ## inplace na dile sometimes it doesn't fill the Nan column

# In[50]:


car_sales_missing


# In[53]:


car_sales


# In[57]:


#Column from series

seats_column= pd.Series([5,5,5,5,5])

# new column called seats

car_sales["Seats"] = seats_column
car_sales


# ## now to fill the NaN data in Seats column

# In[59]:


car_sales["Seats"].fillna(5,inplace=True)
car_sales


# ## Column from Python list
# ## List has to be the same length as the no. of row in csv data

# In[63]:


fuel_economy=[7.5,9.2,5.0,9.6,8.7,9.0,10.1,2.0,5.0,6.0]
car_sales["Fuel Cosumption/100KM"]=fuel_economy
car_sales


# ## Total amount of fuel used by a car = (odometer * fuel consumption per 100km)/100

# In[66]:


car_sales["Used Fuel (L)"]= car_sales["Odometer (KM)"]*car_sales["Fuel Cosumption/100KM"]/100
car_sales


# In[69]:


car_sales.drop("Used Fuel(L)",axis=1,inplace=True)
car_sales


# In[70]:


car_sales.drop("Used Fuel (L)",axis=1,inplace=True)
car_sales


# In[82]:


car_sales.sample(frac=0.2)


# In[88]:


car_sales


# In[90]:


car_sales=pd.read_csv("007 car-sales.csv")
car_sales


# In[91]:


fuel_economy=[7.5,9.2,5.0,9.6,8.7,9.0,10.1,2.0,5.0,6.0]
car_sales["Fuel Cosumption/100KM"]=fuel_economy
car_sales


# In[92]:


car_sales["Used Fuel (L)"]= car_sales["Odometer (KM)"]*car_sales["Fuel Cosumption/100KM"]/100
car_sales


# ## Converting odometer(km) to miles 

# In[94]:


car_sales["Odometer (miles)"]=car_sales["Odometer (KM)"].apply(lambda x: x/1.6)
car_sales


# In[98]:


car_sales.drop("Odometer ()",axis=1,inplace=True)
car_sales


# In[99]:


car_sales["Odometer (miles)"]=car_sales["Odometer (KM)"]
car_sales


# In[100]:


car_sales.rename(columns={'Odometer (KM)':'Odometer (miles)'},inplace=True)
car_sales


# In[101]:


car_sales.drop("Odometer (miles)",axis=1,inplace=True)
car_sales


# In[ ]:




