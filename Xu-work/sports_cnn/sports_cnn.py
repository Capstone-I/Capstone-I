import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

games = pd.read_csv('./archive/games.csv')
details = pd.read_csv('./archive/games_details.csv', low_memory=False)
teams = pd.read_csv('./archive/teams.csv')
players = pd.read_csv('./archive/players.csv')
ranking = pd.read_csv('./archive/ranking.csv')

def get_labels(ranking):
    temp = ranking.copy(deep=True)
    temp = temp.groupby(['TEAM_ID','SEASON_ID'])[['G','W']].max()
    temp = pd.DataFrame(temp)
    temp.reset_index(inplace=True)
    drops = []
    for i in range(len(temp)):
        if temp.iloc[i,1] / 10000 > 2:
            temp.iloc[i,1] = temp.iloc[i,1] % 10000
        else:
            drops.append(i)
            continue;
        if (temp.iloc[i,2] != 82):
            drops.append(i)
    for i in range(len(drops)):
        temp.drop([drops[i]], inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(columns=['index'], inplace=True)
    temp.drop(columns=['G'], inplace=True)
    return temp

def get_features(games, details):
    temp = pd.merge(games, details, how='left', left_on=['GAME_ID'], right_on = ['GAME_ID'])
    temp = temp[['TEAM_ID','SEASON','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
             ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']]
    temp = temp.groupby(['TEAM_ID','SEASON'])[['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']].sum()
    temp = pd.DataFrame(temp)
    next_season = []
    temp.reset_index(inplace=True)
    for i in range(len(temp)):
        next_season.append(temp.iloc[i,1] + 1)
    temp['NEXT_SEASON'] = next_season
    return temp

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

def get_data(ranking, games, details):
    labels = get_labels(ranking)
    features = get_features(games, details)
    data = pd.merge(labels, features, how='left', left_on=['TEAM_ID','SEASON_ID'], right_on = ['TEAM_ID','NEXT_SEASON'])
    data.drop(columns=['SEASON_ID','SEASON'], inplace=True)
    data.dropna(inplace=True)
    data = swap_columns(data, 'W', 'NEXT_SEASON')
    data = data.astype({'NEXT_SEASON': 'int64'})
    data.rename(columns={'W' : 'NEXT_W'}, inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    return data

def scale_data(data):
    temp = data.copy(deep=False)
    std_slc = StandardScaler()
    preprocess = std_slc.fit_transform(temp[['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA'
             ,'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS']])
    data_scaled = pd.DataFrame(preprocess, columns=['FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM'
                    ,'FTA','FT_PCT','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','PLUS_MINUS'])
    data_scaled.insert(0,'TEAM_ID',temp[['TEAM_ID']])
    data_scaled.insert(1,'NEXT_SEASON',temp[['NEXT_SEASON']])
    data_scaled.insert(21,'NEXT_W',temp[['NEXT_W']])
    return data_scaled

def split_data_X_y(data):
    temp = data.copy(deep=False)
    temp.drop(columns=['TEAM_ID','NEXT_SEASON'], inplace=True)
    X = data.iloc[:,2:].copy(deep=False)
    X.drop(columns=['NEXT_W'], inplace=True)
    y = data.iloc[:,-1:].copy(deep=False)
    return X, y


# Prepare data
data = get_data(ranking, games, details)
data_scaled = scale_data(data)
X, y = split_data_X_y(data_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

mlp = MLPRegressor(
    activation='tanh',
    solver='sgd',
    hidden_layer_sizes=(10, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=True,
    max_iter=200,
    shuffle=True
)

mlp.fit(X_train, np.ravel(y_train))
y_pred = mlp.predict(X_test)

def get_2017(data):
    temp = data.copy(deep=False)
    drop_list = []
    for i in range(len(temp)):
        if temp.iloc[i, 1] != 2018:
            drop_list.append(i)
            
    temp.drop(drop_list, inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(columns=['index'], inplace=True)
    return temp

data_2017 = get_2017(data_scaled)
# data_2017

X_2017, y_2018 = split_data_X_y(data_2017)

y_2018_pred = mlp.predict(X_2017)
print(np.round(y_2018_pred))

team_id_name = {}
for i in range(len(teams)):
    team_id_name[teams.iloc[i, 1]] = teams.iloc[i, 5]
print(team_id_name)

name_list = []
for i in range(len(data_2017)):
    name_list.append(team_id_name[data_2017.iloc[i, 0]])
print(name_list)

prediction_dict_2018 = {"team_name": name_list, "wins_pred_2018": np.round(y_2018_pred), 'wins_2018': y_2018['NEXT_W']}
prediction_df_2018 = pd.DataFrame(prediction_dict_2018)
prediction_df_2018.sort_values(by='wins_pred_2018', ascending=False, inplace=True)
prediction_df_2018 = prediction_df_2018.astype({'wins_pred_2018': 'int64'})
prediction_df_2018.reset_index(inplace=True)
prediction_df_2018.drop(columns=['index'], inplace=True)

# prediction_df_2018

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)

# Create a bar graph for actual and predicted wins
plt.figure(figsize=(12, 6))
plt.bar(prediction_df_2018['team_name'], prediction_df_2018['wins_2018'], color='blue', label='Actual Wins', alpha=0.6)
plt.bar(prediction_df_2018['team_name'], prediction_df_2018['wins_pred_2018'], color='red', label='Predicted Wins', alpha=0.6)
plt.xticks(rotation=90)
plt.ylabel('Number of Wins')
plt.title(f'Actual vs Predicted Wins for 2018 Season\nMAE: {mae:.2f} | RMSE: {rmse:.2f}')
plt.legend()
plt.tight_layout()

# Show the graph
plt.show()

prediction_df_2018.to_csv('nba_data.csv', index=False)
