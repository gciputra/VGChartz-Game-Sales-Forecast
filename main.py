#%%
#Shortening column names while keeping them clear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import scipy.stats as stats
from matplotlib.lines import Line2D
import category_encoders as ce



df = pd.read_csv("./Video_Games_Sales_as_at_22_Dec_2016.csv")
#To check whether dataframe works properly: print(df.head())

#To check whether there are null values in csv file: print(df.info()) or dataframe.isnull().sum()

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

#%%
plt.figure(figsize=(10,4))
#Visualizing the distribution of games over the years with a histogram
years_count = df["Year"].max() - df["Year"].min() + 1
plt.hist(df["Year"], bins=years_count, color="mediumspringgreen", edgecolor="black")
plt.title("Games over the Years")
plt.xlabel("Year")
plt.ylabel("Game releases");
plt.show()


# %%
#Correlation between user score and critic score

df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
#Convert data type to float to allow numerical comparation
df[['User_Score', 'Critic_Score']] = df[['User_Score', 'Critic_Score']].astype(float)
user_critic_graph = sns.jointplot(x='User_Score', y='Critic_Score', data = df, color="mediumspringgreen", kind='hex')

#To compute Pearson's correlation coefficient, temporarily remove columns with no pairs of user_score and critic_score
df_temp = df.dropna(subset=['Critic_Score', 'User_Score'])
pearson_result = stats.pearsonr(x=df_temp["User_Score"], y=df_temp["Critic_Score"])
plt.text(5, 10, 'PearsonR: ' + str(pearson_result.statistic))
plt.show()

#%%
#User_Score, Critic_Score, Global Sales per genre
df.groupby("Genre")[["User_Score", "Critic_Score", "Global"]].mean().plot(legend=True, kind="bar")
# %%
def display_missing(data):
    null_count = data.isnull().sum()
    null_percent = (data.isnull().sum()/len(df))*100
    null_table = pd.concat([null_count,null_percent], axis=1)
    null_table.rename(columns={0:'number of null values', 1: 'null value %'}, inplace=True)
    return null_table

display_missing(df)

# %%
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

#%%
df['Regions'] = df[["NA", "EU", "JP", "Other"]].idxmax(1, skipna=True)
sns.countplot(data=df, x='Regions', hue="Has_Scores", palette={
    True: 'lime',
    False: 'tomato'
})
#%%
#Cleaning data for prediction model
df.dropna(subset=['Critic_Score', 'User_Score', 'Rating'], inplace=True)
model_df, outlier_user_counts = rmv_outliers(df, ["Users"])
model_df.drop(columns = ["NA", "EU", "JP", "Other", "Year", "Has_Scores"], inplace=True) #Age is more useful than release year
correlation = model_df.corr(method = 'pearson', numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="YlGnBu")

#%%
numerical = model_df.select_dtypes('number')
categorical = model_df[['Platform', 'Genre', 'Rating']]

encoder = ce.OneHotEncoder()
categorical = encoder.fit_transform(categorical)
features = pd.concat([numerical, categorical], axis=1)
feature_corr = features.corrwith(df["Global"]).dropna().sort_values()
print(features.describe)


#%%
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

#%%
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

# %%
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

# %%

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

#%%
final_err = error(target_test, final_model.predict(features_test))
print("Final Mean Absolute Error of the model is: ", final_err)

#%%
sns.kdeplot(final_model.predict(features_test), label='Prediction', legend=True)
sns.kdeplot(target_test, label='Testing', legend=True)
sns.kdeplot(target_train, label='Training', legend=True)
plt.legend()
plt.ylabel('Density')
plt.xlabel('Global Sales')

#%%

feature_importance = np.argsort(final_model.feature_importances_)
feature_name = features.columns.tolist()
plt.barh(feature_name, feature_importance, color='lime')
plt.ylabel("Features")
plt.title("Relative Importance")
# %%
