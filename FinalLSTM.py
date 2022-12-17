import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

n_hours = 3 #adding 3 hours lags creating number of observations 
n_features = 3 #Features in the dataset.
n_obs = n_hours*n_features

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# LSTM for Price
df1=pd.read_csv('Merge.csv',index_col='DateTime')
df1=df1.iloc[:,[2,0,1]]
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled1 = scaler1.fit_transform(df1.values)
reframed1 = series_to_supervised(scaled1, n_hours, 1)
values1 = reframed1.values
values_last1 = values1[:, -n_features:]
values_X1, values_y1 = values1[:, :n_obs], values1[:, -n_features]
values_X1= values_X1.reshape((values_X1.shape[0], n_hours, n_features))
print(values_X1.shape, values_y1.shape)
model1 = Sequential()
model1.add(LSTM(5, input_shape=(values_X1.shape[1], values_X1.shape[2])))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam')
history1 = model1.fit(values_X1, values_y1, epochs=50, batch_size=6, verbose=2, shuffle=False,validation_split=0.2)


#LSTM for Subjectivity
df2=pd.read_csv('Merge.csv',index_col='DateTime')
df2=df2.iloc[:,[0,1,2]]
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled2 = scaler2.fit_transform(df2.values)
reframed2 = series_to_supervised(scaled2, n_hours, 1)
values2 = reframed2.values
values_last2 = values2[:, -n_features:]
values_X2, values_y2 = values2[:, :n_obs], values2[:, -n_features]
values_X2 = values_X2.reshape((values_X2.shape[0], n_hours, n_features))
print(values_X2.shape, values_y2.shape)
model2 = Sequential()
model2.add(LSTM(5, input_shape=(values_X2.shape[1], values_X2.shape[2])))
model2.add(Dense(1))
model2.compile(loss='mae', optimizer='adam')
history2 = model2.fit(values_X2, values_y2, epochs=50, batch_size=6, verbose=2, shuffle=False,validation_split=0.2)


#LSTM for Polarity
df3=pd.read_csv('Merge.csv',index_col='DateTime')
df3=df3.iloc[:,[1,0,2]]
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaled3 = scaler3.fit_transform(df3.values)
reframed3 = series_to_supervised(scaled3, n_hours, 1)
values3 = reframed3.values
values_last3 = values3[:, -n_features:]
values_X3, values_y3 = values3[:, :n_obs], values3[:, -n_features]
values_X3 = values_X3.reshape((values_X3.shape[0], n_hours, n_features))
print(values_X3.shape, values_y3.shape)
model3 = Sequential()
model3.add(LSTM(5, input_shape=(values_X3.shape[1], values_X3.shape[2])))
model3.add(Dense(1))
model3.compile(loss='mae', optimizer='adam')
history3 = model3.fit(values_X3, values_y3, epochs=50, batch_size=6, verbose=2, shuffle=False,validation_split=0.2)



pred2=list(values_X2[-1][1:])
pred2.append(values_last2[-1])
pred2=(np.array(pred2).reshape((1, n_hours, n_features)))

pred3=list(values_X3[-1][1:])
pred3.append(values_last3[-1])
pred3=(np.array(pred3).reshape((1, n_hours, n_features)))

pred1=list(values_X1[-1][1:])
pred1.append(values_last1[-1])
pred1=(np.array(pred1).reshape((1, n_hours, n_features)))

q=[]
l=[]
# make a prediction
def senti():
	q.append(model2.predict(pred2))

def pol():
	q.append(model3.predict(pred3))


def price():
    global pred1
    global pred2
    global pred3

    yhat1 = model1.predict(pred1)
    inv_yhat1 = np.concatenate((yhat1, (np.array(q)).reshape(1,2)), axis=1)
    del q[0]
    del q[0]

    a=inv_yhat1
    b=[]
    b.append(a[0][2])
    b.append(a[0][1])
    b.append(a[0][0])
    x=[]
    x.append(pred3[0][1])
    x.append(pred3[0][2])
    x.append(b)
    pred3=np.array(x).reshape(1,3,3)

    c=[]
    c.append(a[0][1])
    c.append(a[0][2])
    c.append(a[0][0])
    y=[]
    y.append(pred2[0][1])
    y.append(pred2[0][2])
    y.append(c)
    pred2=np.array(y).reshape(1,3,3)

    z=[]
    z.append(pred1[0][1])
    z.append(pred1[0][2])
    z.append(a[0])
    pred1=np.array(z).reshape(1,3,3)

    inv_yhat1 = scaler1.inverse_transform(inv_yhat1)
    inv_yhat1 = inv_yhat1[:,0]
    l.append(list(inv_yhat1))
	

for i in range(1,12):
			senti()
			pol()
			price()
print(l)
plt.title('Real v Predicted Close_Price')
plt.ylabel('Price ($)')
plt.plot(l, label='Predicted',color='r')
plt.show()