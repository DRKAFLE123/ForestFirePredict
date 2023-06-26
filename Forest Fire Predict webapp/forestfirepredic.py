import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("./Forest Fire Predict webapp/DATA/ForestFireDataCleaned.csv")

# data.loc[:122,'Region']=1  #upto 122 ,Region =1 but intialize as 1.0
# data.loc[122:,'Region']=2  #After 122 , Region = 2 this as 2.0
# data[['Region']] = data[['Region']].astype(int) #1.0 is coverted to 1 astype integer
# #It is used to convert the data in the "Region" column of a pandas DataFrame to integer type.

# data.drop([122,123,168], axis=0, inplace=True)#.reset_index(drop=True)
# data.day.unique()
#  Drop rows at index labels 122 and 123,168

# data.drop([124], axis=0, inplace=True)#.reset_index(drop=True)
# data.columns = data.columns.str.strip()

data[['month', 'day', 'year', 'Temperature', 'RH', 'Ws']] = data[['month', 'day', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

objects = [features for features in data.columns if data[features].dtypes == 'O']  #"o" object type
for i in objects:
    if i != 'Classes':  #exxcept classes
        data[i] = data[i].astype(float)

        # Encoding Not fire as 0 and Fire as 1
# data['Classes']= np.where(data['Classes']== 'not fire',0,1)


data = data.drop(['day','month','year','Ws'], axis=1)

y = data['Classes']
X = data.drop('Classes',axis=1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
del data,X,y


#TRain Function is defined
def train(X_train, y_train, model, hyperparameters):
    grid_search = GridSearchCV(estimator=model,param_grid=hyperparameters, cv = 5)
    grid_search.fit(X_train, y_train)
    
    
    #print the best hyperparameters found
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    # Train the model with best hyperparametres
    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Print the intercept and coefficients of the best model
    # print('Intercept is :', best_model.best_estimator_.intercept_)
    # print('Coefficient is :', best_model.best_estimator_.coef_)

    # Evaluate the best model on the test data
    scores = best_model.score(X_test, y_test)
    print('Score_test_data:', scores)
    
    return best_params, best_model


# EVALUATION

def evaluate_model(X_test, y_test, best_model):
    # it will evaluate the score by taking testing data with best model
        #predict the target values for the best set
    y_pred = best_model.predict(X_test)
    
    
        # Calculate the MSE
    mse = mean_squared_error(y_test, y_pred)
# Calculate the R-squared
    r2 = r2_score(y_test, y_pred)
# Calculate the adjusted R-squared
    # adjusted_r2 = adjusted_r2_score(y_test, y_pred)
# Calculate the MAE
    mae = mean_absolute_error(y_test, y_pred)
# Print the scores
    # print("y test",y_test)
    # print('')
    # print("Prediction",y_pred)
    print("MSE:", mse)
    print("R-squared:", r2)
    # print("Adjusted R-squared:", adjusted_r2)
    print("MAE:", mae)

    return mse,r2,mae


# Define the hyperparameters to tune
hyperparameters = {
    "max_depth": [2,3, 5, 10],
    "min_samples_split": [1,2, 5, 10],
}

# Create a decision tree regressor
model = DecisionTreeRegressor()
_,best_model = train(X_train,y_train,model,hyperparameters)
scores = evaluate_model(X_test,y_test,best_model)   

# from sklearn.linear_model import Ridge
# # Define the hyperparameters to tune
# hyperparameters = {
#     "alpha": np.logspace(-4, 4, 10),
# }

# # Create a Ridge model
# model = Ridge()
# _,best_model = train(X_train,y_train,model,hyperparameters)
# scores = evaluate_model(X_test,y_test,best_model)   
print("Tem,RH,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Region")
inputt=[int(x) for x in [29,57,18,0.0,65.7,7.6,1.3,3.4,0.5,1]]
print(inputt)
final=[np.array(inputt)]

print('')
b = best_model.predict(final)
print("Probability for fire is: ",b[0])


pickle.dump(best_model,open('model.pkl','wb'))
model= pickle.load(open('model.pkl','rb'))

