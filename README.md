# VGChartz Game Sales Forecast Model/EDA
Utilizes Gradient Boosting Regression combined with manual hyperparameter tuning to train and forecast global game sales, as evaluated through mean absolute error.

## Table of contents
[Files and Directory](#files)
[Preparations and Data Cleaning](#prep)
[Data analysis and Visualization](#EDA)
[Regression Model](#model)
[Conclusion](#conclusion)

<a name='files'/>
### Files
- main.py: contains collections of snippets of code for each data visualization
- requirements.txt: list of imported modules and frameworks along with their version
- Video_Games_Sales_as_at_22_Dec_2016.csv: VGChartz 2016 region-specific game sales data and game features data

<a name='prep'/>
### Preparation and Data Cleaning
Python 3 environment comes with many helpful libraries for conducting scientific computing, data handling, and its visualization. Two most prominent modules for data analysis are numpy and pandas as they come with many useful built-in tools
In the following imports: numpy and pandas are used mainly for data grouping and arrangements; matplotlib is used to plot dataframes as visual graph representations along with helpful labels; seaborn adds visual features to the plots; category_encoders will be used later for categorizing data types; lastly, scipy has some useful built-in correlation methods 

Input:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import scipy.stats as stats
from matplotlib.lines import Line2D
import category_encoders as ce
```
Output: 
```
Libraries added to the script! :+1:
```
Now, the dataset file (in csv form) is called and converted into a dataframe with the pandas library. For future ease of use and efficiency purposes, I will also rename and shorten column headers while still keeping each column uniquely identifiable. Afterwards, I will conduct an initial exploration of the dataset before performing any analysis. 
I also added a new column named age which computes the current year (2023) with each game's release year in the column 'Years'.
Input:
```
df = pd.read_csv("./Video_Games_Sales_as_at_22_Dec_2016.csv")
#To check whether dataframe works properly: print(df.head())

#To check whether there are null values in csv file: print(df.info()) or dataframe.isnull().sum()

#Shortening column names while keeping them clear
df.rename(columns={
    'Year_of_Release':'Year', 
    'Global_Sales':'Global',
    'NA_Sales': 'NA',
    'EU_Sales': 'EU',
    'JP_Sales': 'JP',
    'Other_Sales': 'Other',
    'Critic_Count': 'Critics',
    'User_Count': 'Users'}, inplace=True)

#Removing games which contain null values for essential features such as critic score or year of release
df.dropna(subset=["Year", "Global", "Genre","NA", "EU", "JP", "Other"],inplace=True)

#Adding a new column named age
df['Year'] = df['Year'].apply(int)
df['Age'] = 2023 - df['Year']

print(df.describe(include='all'))
```
Output:
```
Name Platform          Year   Genre  \
count                         16448    16448  16448.000000   16448   
unique                        11429       31           NaN      12   
top     Need for Speed: Most Wanted      PS2           NaN  Action   
freq                             12     2127           NaN    3308   
mean                            NaN      NaN   2006.488996     NaN   
std                             NaN      NaN      5.877470     NaN   
min                             NaN      NaN   1980.000000     NaN   
25%                             NaN      NaN   2003.000000     NaN   
50%                             NaN      NaN   2007.000000     NaN   
75%                             NaN      NaN   2010.000000     NaN   
max                             NaN      NaN   2020.000000     NaN   

              Publisher            NA            EU            JP  \
count             16416  16448.000000  16448.000000  16448.000000   
unique              579           NaN           NaN           NaN   
top     Electronic Arts           NaN           NaN           NaN   
freq               1344           NaN           NaN           NaN   
mean                NaN      0.263965      0.145895      0.078472   
std                 NaN      0.818286      0.506660      0.311064   
min                 NaN      0.000000      0.000000      0.000000   
25%                 NaN      0.000000      0.000000      0.000000   
50%                 NaN      0.080000      0.020000      0.000000   
75%                 NaN      0.240000      0.110000      0.040000   
max                 NaN     41.360000     28.960000     10.220000   

               Other       Global  Critic_Score      Critics User_Score  \
count   16448.000000  16448.00000   7983.000000  7983.000000       9840   
unique           NaN          NaN           NaN          NaN         96   
top              NaN          NaN           NaN          NaN        tbd   
freq             NaN          NaN           NaN          NaN       2377   
mean        0.047583      0.53617     68.994363    26.441313        NaN   
std         0.187984      1.55846     13.920060    19.008136        NaN   
min         0.000000      0.01000     13.000000     3.000000        NaN   
25%         0.000000      0.06000     60.000000    12.000000        NaN   
50%         0.010000      0.17000     71.000000    22.000000        NaN   
75%         0.030000      0.47000     79.000000    36.000000        NaN   
max        10.570000     82.53000     98.000000   113.000000        NaN   

               Users Developer Rating           Age  
count    7463.000000      9907   9769  16448.000000  
unique           NaN      1680      8           NaN  
top              NaN   Ubisoft      E           NaN  
freq             NaN       201   3922           NaN  
mean      163.015141       NaN    NaN     16.511004  
std       563.863327       NaN    NaN      5.877470  
min         4.000000       NaN    NaN      3.000000  
25%        10.000000       NaN    NaN     13.000000  
50%        24.000000       NaN    NaN     16.000000  
75%        81.000000       NaN    NaN     20.000000  
max     10665.000000       NaN    NaN     43.000000  
```
Immediately, it is highlighted that the dataset has numerous flaws such as insufficient data, incorrect data types, and null values. For example, there are ~17K game titles but only ~8K game titles has a reported Critic_Score and ~10K has a User_Score. That is more than 50% of insufficient data! I will handle this later, and focus on more pressing issues such as User_Score containing object type data instead of float.  
Input:
```
#Convert String to Float
df[['User_Score', 'Critic_Score']] = df[['User_Score', 'Critic_Score']].astype(float)
```
Output:
```
The dataset is now mostly ready for data visualization and EDAs! :+1:
```
 
<a name='eda'/>
### Exploratory Data Analysis and Visualization
The dataset contains many interesting features such as genre, rating, developer, etc. The purpose of this section is to summarize interesting correlations between features and gain a deeper understanding of how game features affect each other before jumping to regression models. 

Input:
```
plt.figure(figsize=(10,4))
#Visualizing the distribution of games over the years with a histogram
years_count = df["Year"].max() - df["Year"].min() + 1
plt.hist(df["Year"], bins=years_count, color="mediumspringgreen", edgecolor="black")
plt.title("Games over the Years")
plt.xlabel("Year")
plt.ylabel("Game releases");
plt.show()
```
Output:

![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/019d9270-6eb3-4c8a-9606-54513ebbcf47)
Here, I discovered that peak gaming release counts is around 2005-2010, increasing sharply from 1985 and decreasing slowly afterwards. It symbolizes the rapid growth of the gaming industry as we enter the 21st Century, with more companies and franchises introduced before slowing down after 2010. 

Now, I want to know the correlation between user and critic scores. But there are 2 problems which will be discussed below:
1st, more than 50% missing values in user and critic scores! 
Input:
```
def display_missing(data):
    null_count = data.isnull().sum()
    null_percent = (data.isnull().sum()/len(df))*100
    null_table = pd.concat([null_count,null_percent], axis=1)
    null_table.rename(columns={0:'number of null values', 1: 'null value %'}, inplace=True)
    return null_table

display_missing(df)
```
Output:
|number of null values| null value %|
| ----- | ------ | 
| Publisher	| 32 | 0.194553 |
| Critic_Score | 8465 |	51.465224 |
| Critics	| 8465 | 51.465224 |
| User_Score | 8985 |	54.626702 |
| Users | 8985 | 54.626702 |
| Developer	| 6541 | 39.767753 |
| Rating | 6679 |	40.606761 |

Normally, this would be a sign to remove the columns entirely due to insufficiency but I thought that critic and user scores would be invaluable data for sales prediction. In fact, I usually look at reviews first before purchasing video games. To evaluate my decision, I decided to look at the game releases which do not have either user or critic scores over the years:  
Input:
```
df['Has_Scores'] = df['User_Score'].notnull() & df['Critic_Score'].notnull()

plt.hist(df[df["Has_Scores"]==True]["Year"], color="lime", alpha=0.5, 
         bins=range(1980, 2021), edgecolor="black")
plt.hist(df[df["Has_Scores"]==False]["Year"], color="tomato", alpha=0.5, 
         bins=range(1980, 2021), edgecolor="black")
plt.title('Games over the Years')
plt.xlabel("Release Year")
plt.ylabel("Game Count")
plt.legend(handles=[Line2D([0], [0], color="lime", lw=20, label="True", alpha=0.5),
                    Line2D([0], [0], color="tomato", lw=20, label="False", alpha=0.5)],
           title="Has Critic and User Score?", loc=6);
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/18791875-3dc5-464a-b36d-09f0bb7d2b8f)

Since, most games without critic and user scores are from the older generation, I thought that it wouldn't be as impactful for my prediction model and thus, the optimum solution is to drop game entries entirely if they are missing even one of either user scores or critic scores.
I discovered another problem: the top value in the User_Score column is "tbd" which I first need to convert to np.nan for the program to register it as an empty value.
Input:
```
#Convert data type to float to allow numerical comparation
df['User_Score'] = df['User_Score'].replace('tbd', np.nan)

user_critic_graph = sns.jointplot(x='User_Score', y='Critic_Score', data = df, color="mediumspringgreen", kind='hex')

#To compute Pearson's correlation coefficient, temporarily remove columns with no pairs of user_score and critic_score
df_temp = df.dropna(subset=['Critic_Score', 'User_Score'])
pearson_result = stats.pearsonr(x=df_temp["User_Score"], y=df_temp["Critic_Score"])
plt.text(5, 10, 'PearsonR: ' + str(pearson_result.statistic))
plt.show()
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/1cbb85b3-0088-4c91-b7a2-e86b24c83943)

Most of the games generally have great scores of 7+/10 from users and 70+/100 for critics. There is also a positive but weak correlation of 0.5797 between critic scores and user scores where users typically give more lower scores than critics, which I find interesting. 

Next, I wanted to check the user scores and critic scores per genre to see how each field is performing in ratings. 
Input:
```
#User_Score, Critic_Score, Global Sales per genre
df.groupby("Genre")[["User_Score", "Critic_Score", "Global"]].mean().plot(legend=True, kind="bar")
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/f4040371-18ef-4d2e-a69e-89fa7f8ec44b)

Suprisingly, despite some genres being more popular such as Shooters and some more hidden such as Puzzle games, the ratings are generally equal at about 7 for user scores and 60+ for critic scores. 
Next, I need to resolve the extreme outliers in user counts and global sales. For user counts, the outliers are when the games have very few users reporting a user score. I thought the opinions of a few individuals would be unreliable, thus I removed game entries with extremely few user counts. 
For Global Sales, it was harder to decide since outliers could be useful to the sales forecast since they represent the best-sellers, therefore to balance the trade-off between accuracy and data variability, I only removed extreme outliers and kept mild outliers.
Input:
```
#Cleaning data for prediction model
def rmv_outliers(data, key_list):
    df_out = data
    for key in key_list:
        #Compute the first and third quartile, and interquartile range
        f_quartile = df_out[key].describe()["25%"]
        t_quartile = df_out[key].describe()['75%']
    iqr = t_quartile - f_quartile
    #Removing and isolating extreme outliers
    outliers = df_out[(df_out[key] <= (f_quartile - 3*iqr)) | (df_out[key] >= (t_quartile + 3*iqr))]
    df_out = df_out[(df_out[key] > (f_quartile - 3*iqr)) & (df_out[key] < (t_quartile + 3*iqr))]
    return df_out, outliers
df, global_outlier = rmv_outliers(df, ['Global'])
df['Has_Scores'] = df['User_Score'].notnull() & df['Critic_Score'].notnull()
df.dropna(subset=['Critic_Score', 'User_Score', 'Rating'], inplace=True)
model_df, outlier_user_counts = rmv_outliers(df, ["Users"])
```
As a final summary of all the game features, I plotted a correlation graph between each numerical feature before building the regression model.
Input:
```
model_df.drop(columns = ["NA", "EU", "JP", "Other", "Year", "Has_Scores"], inplace=True) #Age is more useful than release year
correlation = model_df.corr(method = 'pearson', numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="YlGnBu")
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/7e73dbda-95d5-4165-977b-acd862cf6dde)

<a name='model'/>
### Picking the Model and Optimization
First, I categorized the features between numerical and categorical and utilized one hot encoding to assign numerical representations to categorical values to perform correlation analysis with global sales.
Input:
numerical = model_df.select_dtypes('number')
categorical = model_df[['Platform', 'Genre', 'Rating']]

encoder = ce.OneHotEncoder()
categorical = encoder.fit_transform(categorical)
features = pd.concat([numerical, categorical], axis=1)
feature_corr = features.corrwith(df["Global"]).dropna().sort_values()
feature_corr.tail(5)
feature_corr.head(5)

Output:
| User_Score | 0.155011 |
| Users | 0.243130 |
| Critic_Score | 0.280368 |
| Critics | 0.293808 |
|Global | 1.000000 |

| Platform_7 | -0.188800 |
| Genre_6 | -0.089523 |
| Genre_10 |-0.086620 |
| Rating_6 | -0.066765 |
| Platform_10 | -0.064360 |
Highest correlation are critic and user scores and counts which are understandable due to their importance in affecting customer judgements, but the lowest correlation being the PC_Platform was surprising. 

Next, I wanted a control model as a basis to evaluate the performance of more complex regression models. For this, I chose the median sales as the baseline predictor and picked mean absolute error rather than RMSE as the measurement of accuracy or the cost function since there are still some mild outliers. 
Input:
```
from sklearn.model_selection import train_test_split
target = pd.Series(features['Global'])
features.drop(columns='Global', inplace=True)
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)
#Absolute error calculator 
def error(true, pred):
    return np.average(abs(true - pred))

baseline = np.median(target_train)
baseline_error = error(target_test, baseline)

print("Baseline Global Sales Prediction: ", baseline)
print("Baseline Guess Error: ", baseline_error)
```
Output:
```
Baseline Global Sales Prediction:  0.23
Baseline Guess Error:  0.2787353206865403
```
Next, I used a universal standard function to fit and evaluate the training data with several sklearn built-in regression models including Linear Regression, Ridge, RFR, GBR, SVR, and KNR to pick the best model to hypertune later on. 
Input:
```
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def evaluate(model):
    #training
    model.fit(features_train, target_train)

    #prediction and error
    model_pred = model.predict(features_test)
    model_err = error(target_test, model_pred)

    return model_err

#prediction models
model_name = ['LR', 'RFR', 'GBR', 'SVR', 'Ridge', 'KNR']
models = [LinearRegression(), RandomForestRegressor(random_state=60), GradientBoostingRegressor(random_state=60), SVR(C=1000, gamma=0.1), Ridge(alpha=10), KNeighborsRegressor(n_neighbors=10)]

model_err_list = []
for item in models:
   model_err_list.append(evaluate(item))

plt.bar(model_name, model_err_list, color='lime', edgecolor = 'black')
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/72ab209a-74fb-4a22-9638-59b3620625a3)

It seems the model with least arror is Gradient Boosting Regressor followed closely by Random Forest, and the worst model is SVR. I will then continue using GBR. According to the GBR SKlearn documentation, [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) the hyperparameters to tune our model is the loss function, max_depth, min_samples_leaf, min_samples_split, and max_features. Since the hyperparameter space is big, I will use random search cross-validation for efficiency purposes. 
Input:
```
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#Testing loss functions
loss = ["squared_error", "absolute_error", "huber", "quantile"]
#maximum depth of tree
max_depth = [3,4,5,6,7,8]
#minimum samples per leaf
min_samples_leaf = [1,2,4,6,8]
#minimum samples to split a node
min_samples_split = [2,4,6,8,10]
#maximum no. of features to consider to make splits
max_features = ['sqrt', 'log2']

hyperparameter_dict = {
    'loss': loss,
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'max_features': max_features,

}

random_cv = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=hyperparameter_dict,
    cv=4,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    verbose=1,
    return_train_score=True,
    random_state=42
)

random_cv.fit(features_train, target_train)
print(random_cv.best_estimator_)
```
Output:
```
Fitting 4 folds for each of 20 candidates, totalling 80 fits
GradientBoostingRegressor(loss='absolute_error', max_depth=7,
                          max_features='sqrt', min_samples_leaf=8,
                          min_samples_split=8, random_state=42)
```
Finally, I used Grid Search to find the optimal number of n_estimators and since there is only one dimension, it would be better to use grid search without sacrificing significant run-time. 
Input:
```
num_trees_dict = {
    "n_estimators": [50, 100, 150, 200, 250, 300]
}

grid_search = GridSearchCV(
    estimator=random_cv.best_estimator_,
    param_grid=num_trees_dict,
    cv = 4,
    scoring='neg_mean_absolute_error',
    verbose=1,
    return_train_score=True
)
grid_search.fit(features_train, target_train)

final_model = grid_search.best_estimator_
print(final_model)
```
Output:
```
Fitting 4 folds for each of 6 candidates, totalling 24 fits
GradientBoostingRegressor(loss='absolute_error', max_depth=7,
                          max_features='sqrt', min_samples_leaf=8,
                          min_samples_split=8, n_estimators=300,
                          random_state=42)
```
<a name='conclusion'/>
### Data Summary
Our final prediction model achieves an error of:
Input:
```
final_err = error(target_test, final_model.predict(features_test))
print("Final Mean Absolute Error of the model is: ", final_err)
```
Output:
```
Final Mean Absolute Error of the model is:  0.20647691365259413
```

This graph shows that the prediction density is consistently shifted to the right of the actual values. This can be solved with further tuning in the future. 
Input:
```
sns.kdeplot(final_model.predict(features_test), label='Prediction', legend=True)
sns.kdeplot(target_test, label='Testing', legend=True)
sns.kdeplot(target_train, label='Training', legend=True)
plt.legend()
plt.ylabel('Density')
plt.xlabel('Global Sales')
```
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/c831f03f-2a31-497d-99e2-6303119ecff0)
These are the features that are most prominent in affecting game sales. 
Input:
feature_importance = np.argsort(final_model.feature_importances_)
feature_name = features.columns.tolist()
plt.barh(feature_name, feature_importance, color='lime')
plt.ylabel("Features")
plt.title("Relative Importance")
Output:
![image](https://github.com/gciputra/VGChartz-Game-Sales-Forecast/assets/140233515/8432e062-a6f3-49fc-a3b7-8a7242f12cc0)

Guide:
Platforms: ['PS2', 'GBA', 'X360', 'PS3', 'PC', 'Wii', 'PSP', 'PS', 'XB', 'GC','DS', 'XOne', '3DS', 'DC', 'PS4', 'WiiU', 'PSV']
Genre: ['Shooter', 'Action', 'Role-Playing', 'Racing', 'Simulation', 'Sports', 'Fighting', 'Platform', 'Misc', 'Strategy', 'Puzzle', 'Adventure']
Ratings: ['M', 'E', 'T', 'E10+', 'RP']
