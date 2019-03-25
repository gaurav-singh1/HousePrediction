
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(6,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def wider_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compiling model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



#load dataset

dataframe = pandas.read_csv('/Users/aayush/Documents/FarmguideProjects/TensorFlow/ConvNetTraining/src/BostonHousePricePrediction/data/housing.data',
                            delim_whitespace=True, header=None)

dataset = dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]

seed=7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100,
                           batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold)

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



# # standardizing the dataset
#
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model,
#                                          epochs=50, batch_size=5,verbose=0)))
# pipeline = Pipeline(estimators)
# # kfold = KFold(n_splits=10, random_state=seed)
# results=cross_val_score(pipeline, X,Y,cv=kfold)
# print("Standardized: %.2f (%.2f) MSE " % (results.mean(), results.std()))



# standardizing dataset with wider model, 13->20->1

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100
                                        , batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

kfold=KFold(n_splits=10, random_state=seed)
results=cross_val_score(pipeline, X,Y,cv=kfold)

print("Wider : %.2f (%.2f) MSE " % (results.mean(), results.std()))