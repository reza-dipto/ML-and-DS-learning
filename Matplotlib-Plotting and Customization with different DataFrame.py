#!/usr/bin/env python
# coding: utf-8

# # Introduction to Matplotlib

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


plt.plot();


# In[3]:


plt.plot();
plt.show()


# In[4]:


plt.plot([1,2,3,4]);


# In[5]:


x=[1,3,6,7]
y=[11,12,33,34]
plt.plot(x,y)


# In[6]:


plt.plot();


# In[7]:


#2nd method
fig = plt.figure()# it creates a figure
ax= fig.add_axes([1,1,1,1])
ax.plot(x,y) #adding data
plt.show()


# In[8]:


# 3rd and recommended method

fig ,ax=plt .subplots()
ax.plot(x,[50,100,200,250]);


# In[9]:


type(fig),type(ax)


# # Matplotlib Example workflow

# In[10]:


fig


# In[11]:


fig = plt.figure()


# In[12]:


fig


# In[13]:


ax=plt.subplots(figsize=(20,20))


# In[14]:


ax


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#1. Prepare data
x=[1,2,3,4]
y=[11,16,21,26]

#.2 Set the plot
fig ,ax= plt.subplots(figsize=(5,10))

#3 plot the data
ax.plot(x,y)

#4 Customize plot
ax.set(title="Basic Plot",
       xlabel="x-axis",
       ylabel="y-axis")
        
#5 Save

fig.savefig("Images/basic-plot.png")


#   # Making figures with numpy arrays
#   
#   * Line plot
#   * scatter plot
#   * barplot
#   * histogram
#   * subplot

# In[16]:


import numpy as np


# In[17]:


# Create data
   
x= np.linspace(0,10,100) 
x[:10]


# In[18]:


# plot the data
  
fig , ax= plt.subplots()
ax.plot(x,x**2);


# In[19]:


# use same data to make scatter plot

fig, ax= plt.subplots()
ax.scatter(x,np.exp(x))


# In[20]:


fig, ax= plt.subplots()
ax.scatter(x,np.sin(x));


# In[21]:


# make a plot from dictionary

nut_butter_prices ={"Almond":10,
                   "Peanut":8,
                   "Cashew":12}
fig,ax=plt.subplots()
ax.bar(nut_butter_prices.keys(),nut_butter_prices.values())
ax.set(title="Mayer Dowa Store",
        ylabel="price ($)");


# In[22]:


# horizontal bar graph
    
fig,ax=plt.subplots()
ax.barh(list(nut_butter_prices.keys()),list(nut_butter_prices.values()));


# In[23]:


# Histogram


# In[24]:


# maing data

x= np.random.randn(1000)
fig , ax=plt.subplots()
ax.hist(x);


# ## Two options for Subplots

# In[25]:


#option 1

fig, ((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(10,5))

# plot to each different axis

ax1.plot(x,x/2);
ax2.scatter(np.random.random(10),np.random.random(10));
ax3.bar(nut_butter_prices.keys(),nut_butter_prices.values());
ax4.hist(np.random.randn(1000));


# ### SEE ANATOMY OF A MATPLOTLIB PLOT PNG FILE

# In[26]:


# SUbplot option 2


# In[27]:


fig, ax=plt.subplots(nrows=2,
                    ncols=2,
                    figsize=(10,5))

# Now plot to each diff index

ax[0,0].plot(x,x/2);
ax[0,1].scatter(np.random.random(10),np.random.random(10));
ax[1,0].bar(nut_butter_prices.keys(),nut_butter_prices.values());
ax[1,1].hist(np.random.randn(1000));


#  ## Plotting from pandas dataframe

# In[28]:


import pandas as pd


# In[29]:


# making a dataframe


# In[30]:


car_sales = pd.read_csv("007 car-sales.csv")
car_sales


# In[31]:


# copied library code
n [3]: np.random.seed(123456)

In [4]: ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))

In [5]: ts = ts.cumsum()

In [6]: ts.plot();


# In[ ]:


ts = pd.Series(np.random.randn(1000), 
               index=pd.date_range("1/1/2023",
               periods=1000)) ;

ts=ts.cumsum()
ts.plot();


# In[ ]:


car_sales


# In[32]:


car_sales["Price"]= car_sales["Price"].str.replace('[\$\,\.]','')
car_sales


# In[33]:


#remove the last two zeros

car_sales["Price"]= car_sales["Price"].str[:-2]
car_sales



# In[34]:


car_sales["Sale Date"]=pd.date_range("1/1/2020",periods=len(car_sales))


# In[35]:


car_sales


# In[36]:


car_sales["Total Sales"]=car_sales["Price"].cumsum()
car_sales


# In[37]:


# Total sales gave a junk value beacuse the price column is in string format

type(car_sales["Price"][0])


# In[38]:


#so type casting the price column to int 


# In[39]:


car_sales["Total Sales"]=car_sales["Price"].astype(int).cumsum()
car_sales


# In[40]:


#plotting the total sales

car_sales.plot(x="Sale Date",y="Total Sales");


# In[41]:


# scatter plot
car_sales["Price"]=car_sales["Price"].astype(int) # because scatter plot y value int hobe

car_sales.plot(x="Odometer (KM)",y="Price",kind="scatter");


# In[42]:


#bar graph

x= np.random.rand(10,4)
x


# In[43]:


# making x as dataframe
df=pd.DataFrame(x,columns=['a','b','c','d'])
df


# In[44]:


df.plot.bar(); # also works as df.plot(kind="bar"); 


# In[45]:


car_sales


# In[46]:


car_sales.plot(x="Make",y="Odometer (KM)",kind="bar");


# In[47]:


# histogram

car_sales["Odometer (KM)"].plot.hist();

## also works as car_sales["Odometer (KM)"].plot(kind="hist");


# In[48]:


# another dataset


# In[49]:


heart_disease = pd.read_csv("011 heart-disease.csv")
heart_disease


# In[50]:


heart_disease.head()


# In[51]:


# creating a histogram to see the distribution of age column

heart_disease["age"].plot.hist(bins=100);


# In[52]:


heart_disease.head()


# In[53]:


heart_disease.plot.hist(figsize=(10,60),subplots=True);


#   # pyplot vs matplotlib oo method
#     
#     * When plotting something quicly , use pyplot method
#     * When plotting something more advanced , use object oriented method
# 

# In[54]:


heart_disease.head()


# In[55]:


over_50 = heart_disease[heart_disease["age"]>50]
over_50.head()


# In[56]:


len(over_50)


# In[57]:


over_50.plot(kind='scatter',
            x='age',
            y='chol',
            c='target');


# In[58]:


## object oriented method

fix, ax=plt.subplots(figsize=(10,15))

over_50.plot(kind='scatter',
            x='age',
            y='chol',
            c='target',
            ax=ax);
# setting x limit a bit wider

ax.set_xslim([45,100]) 


# In[ ]:


# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);


# In[ ]:


over_50.target.values


# In[ ]:


# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);


# In[ ]:


# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);
    
#customize
ax.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");


# In[ ]:


# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);
    
#customize
ax.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");       

# Add a legend
ax.legend(*scatter.legend_elements(),title="target");


# In[59]:


# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);
    
#customize
ax.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");       

# Add a legend
ax.legend(*scatter.legend_elements(),title="target");

# adding a horizontal line in matplotlib plot

ax.axhline(over_50["chol"].mean(),
            linestyle='--');


# In[60]:


over_50.head()


# In[64]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])


# In[80]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   


# In[81]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");


# In[82]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');


# In[83]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"])


# In[84]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"])

#customize ax1
ax1.set(title="Heart Disesse and Max heart rate",
       xlabel="age",
       ylabel="Max heart rate");


# In[85]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10))

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
       xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"])

#customize ax1
ax1.set(title="Heart Disesse and Max heart rate",
       xlabel="age",
       ylabel="Max heart rate");

#add a legend to ax1
ax1.legend(*scatter.legend_elements(),title="target");


# In[87]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10),
                            sharex=True)

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
      # xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"])

#customize ax1
ax1.set(title="Heart Disesse and Max heart rate",
       xlabel="age",
       ylabel="Max heart rate");

#add a legend to ax1
ax1.legend(*scatter.legend_elements(),title="target");

#add a meanline to ax1

ax1.axhline(y=over_50["thalach"].mean(),
            linestyle='--');


# In[89]:


#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10),
                            sharex=True)

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"])

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
      # xlabel ="Age",
       ylabel="Cholesterol");   

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"])

#customize ax1
ax1.set(title="Heart Disesse and Max heart rate",
       xlabel="age",
       ylabel="Max heart rate");

#add a legend to ax1
ax1.legend(*scatter.legend_elements(),title="target");

#add a meanline to ax1

ax1.axhline(y=over_50["thalach"].mean(),
            linestyle='--');

## add a title to the figure
fig.suptitle("Heart Disease Analysis",fontsize=16,fontweight="bold");


# # customize matplotlib plots
# 

# In[90]:


plt.style.available


# In[92]:


car_sales.head()


# In[93]:


car_sales["Price"].plot();


# In[94]:


plt.style.use('seaborn-whitegrid') 


# In[95]:


car_sales["Price"].plot();


# In[96]:


plt.style.use('seaborn');


# In[97]:


car_sales["Price"].plot()


# In[98]:


car_sales.plot(x="Odometer (KM)",y="Price",kind="scatter");


# In[100]:


plt.style.use('ggplot')
car_sales["Price"].plot();


# In[101]:


# create dummy data for titles and legends


# In[103]:


x= np.random.randn(10,4)
x


# In[104]:


df= pd.DataFrame(x,columns=['a','b','c','d'])


# In[105]:


df


# In[106]:


ax= df.plot(kind='bar')
type(ax)


# In[109]:


#customize the plot with the set() method
ax= df.plot(kind = "bar")

# add label and a title
ax.set (title="Random number bat graph from dataframe",
       xlabel="Row number",
       ylabel ="Random number")

# make the legend visible
ax.legend().set_visible(True)


# In[115]:


# set the style 
   
plt.style.use('seaborn-whitegrid')

# oo mehtod from scratch

fig, ax= plt.subplots(figsize=(10,6))

# plot the data
scatter = ax.scatter(x=over_50["age"],
                    y=over_50["chol"],
                    c=over_50["target"],
                   cmap="plasma"); # this changes the color scheme
   
#customize
ax.set(title= "Heart Disease and Chol Levels",
      xlabel ="Age",
      ylabel="Cholesterol");       

# Add a legend
ax.legend(*scatter.legend_elements(),title="target");

# adding a horizontal line in matplotlib plot

ax.axhline(over_50["chol"].mean(),
           linestyle='--'); 


# In[124]:


# customizing the y and x axis limitations

#subpolot of chol ,age , thalach
fig, (ax0,ax1)= plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10,10),
                            sharex=True)

#add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"],
                     cmap="winter")

#Customize ax0
ax0.set(title= "Heart Disease and Chol Levels",
      # xlabel ="Age",
       ylabel="Cholesterol");   

ax0.set_xlim([50,80]) #change ax0 x axis limits

# Add a legend ax0

ax0.legend(*scatter.legend_elements(),title="target");

# adding a meanline

ax0.axhline(y=over_50["chol"].mean(),
            linestyle='--');

# add data to ax1
scatter = ax1.scatter(x=over_50["age"] ,
                     y=over_50["thalach"],
                     c=over_50["target"],
                     cmap="winter")

#customize ax1
ax1.set(title="Heart Disesse and Max heart rate",
       xlabel="age",
       ylabel="Max heart rate");

ax1.set_xlim([50,80])#change ax1 x axis limits
ax1.set_ylim([60,200]) #change ax1 y axis limits


#add a legend to ax1
ax1.legend(*scatter.legend_elements(),title="target");

#add a meanline to ax1

ax1.axhline(y=over_50["thalach"].mean(),
            linestyle='--');

## add a title to the figure
fig.suptitle("Heart Disease Analysis",fontsize=16,fontweight="bold");


# In[ ]:




