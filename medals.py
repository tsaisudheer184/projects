import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
teams=pd.read_csv('teams.csv')
teams=teams[['team','country','year','athletes','age','prev_medals','medals']]
teams
teams.corr(numeric_only=True)["medals"]
""" the correlation between medals and athletes is 0.84 which is good and the correlation between the prev_medals and medals is 0.92 
which is very much helpful 
"""
"""plots of medals v/s athletes 
   plots of medals v/s age """
sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True)
sns.lmplot(x="medals",y="age",data=teams,fit_reg=True,ci=None)
""" histogram of Medals """
teams.plot.hist(y='medals')
""" find The row having null values """
teams[teams.isnull().any(axis=1)]
""" Cleaning data by removing all the row having null values """
teams=teams.dropna()
""" splitting Data  into train and test because we will train the model based  on the previous data up to 2012 and test on data from 2012 
this how we can predict the working of our model"""
train=teams[teams['year'] < 2012].copy()
test=teams[teams['year'] >= 2012].copy()
reg=LinearRegression() #using Linear Regression
predictors=['athletes','prev_medals']
target="medals"
predictions=reg.predict(test[predictors])
test["predictions"]=predictions
test.loc[test["predictions"] < 0,"predictions"] = 0 
test["predictions"]=test["predictions"]
error=mean_absolute_error(test["medals"],test["predictions"])
error
""" this means that we were 3.3 medals + or - on how many actula medals """
"""checking"""
teams.describe()["medals"]
test[test["team"] == "USA"]
""" actually in 2012 usa got 248 medals but we predicted 285 and similarly in 2016 """ 
# finding Errors by each Team
errors = (test['medals']-test['predictions']).abs()
#error mean by each team
errors_by_each_team=errors.groupby(test["team"]).mean()
print("errors while Predictin : ",errors_by_each_team)
#finding avg medals won by each team
medals_by_team=test["medals"].groupby(test["team"]).mean()
error_ratio=errors_by_each_team/medals_by_team
error_ratio=error_ratio[~pd.isnull(error_ratio)]
error_ratio=error_ratio[np.isfinite(error_ratio)]
print("ratio to the error and actual ",error_ratio)
error_ratio['IND']
