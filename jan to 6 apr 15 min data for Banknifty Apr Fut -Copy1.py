#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[2]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(25,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[3]:


import pandas as pd
df=pd.read_csv('NIFTY BANK (20230125104500000 _ 20221114113000000) - NIFTY BANK (20230125104500000 _ 20221114113000000).csv')
df.head()


# In[4]:


df.tail()


# In[5]:


# df = df.drop(index=df.tail(1).index)


# In[6]:


df.tail()


# In[7]:


df = pd.DataFrame(df)


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

df2=df.reset_index()['Close']
import matplotlib.pyplot as plt
plt.plot(df2[1000:])


# In[9]:


df2.shape


# In[10]:


len(df2)


#  ## Run from here

# In[11]:


df2 = df2.drop(df.index[-1])


# In[12]:


# df2.loc[1274] = 41900   #next value to be entered



# In[13]:


# df2 = pd.DataFrame(df2)


# In[14]:


df2.shape


# In[15]:


df2.tail()


# In[16]:


# df2 = df2.drop(index=df2.tail(1).index)


# In[17]:


df2.tail()


# In[18]:


plt.figure(figsize=(10,10))

# df2=df.reset_index()['Close']
import matplotlib.pyplot as plt
plt.plot(df2)


# In[19]:


import numpy as np
df2


# In[20]:


df1 = pd.DataFrame(df2)


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1)


# In[22]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
training_size,test_size


# In[23]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[24]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 25
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[25]:


print(X_train.shape), print(y_train.shape)


# In[26]:


print(X_test.shape), print(ytest.shape)


# In[27]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[28]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[ ]:


loss=pd.DataFrame(model.history.history)


# In[ ]:


plt.figure(figsize=(10,10))
loss.plot()
plt.show()


# In[ ]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[ ]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[ ]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[ ]:


### Plotting 
# shift train predictions for plotting
plt.figure(figsize=(10,10))
look_back=25
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


len(test_data)
# subtract number from this to get the output = step


# In[ ]:


x_input=test_data[493:].reshape(1,-1)
x_input.shape


# In[ ]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[ ]:


temp_input


# In[ ]:


len(temp_input)


# In[ ]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=25
i=0
while(i<2):
    
    if(len(temp_input)>25):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[ ]:


scaler.inverse_transform(lst_output)


# In[ ]:


scaler.inverse_transform(lst_output)
# predicted value


# In[ ]:


day_new=np.arange(1,3)
day_pred=np.arange(3,5)
import matplotlib.pyplot as plt
len(df1)


# In[ ]:


plt.plot(day_new,scaler.inverse_transform(df1[2589:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[ ]:


plt.figure(figsize=(10,10))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[100:])


# In[ ]:


plt.figure(figsize=(10,10))

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df1[100:])


# In[ ]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=80
i=0
while(i<1):
    
    if(len(temp_input)>80):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[ ]:


scaler.inverse_transform([[1.0473953485488892]])


# In[ ]:


len(lst_output)


# In[ ]:


day_new=np.arange(1,201)
day_pred=np.arange(201,401)
import matplotlib.pyplot as plt
len(df1)


# In[ ]:


plt.plot(day_new,scaler.inverse_transform(df1[875:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[ ]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[300:])


# In[ ]:


df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)


# In[ ]:




