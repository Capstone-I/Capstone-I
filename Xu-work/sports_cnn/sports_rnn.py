import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Neural Network Architecture
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Initialize neural network, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = (256, 128)
output_size = 1
model = Net(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(200):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Testing
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).flatten()
    test_loss = criterion(test_predictions, y_test_tensor.flatten())

print(f"Test Loss (MSE): {test_loss.item()}")


y_test_np = y_test_tensor.numpy().flatten()
test_predictions_np = test_predictions.numpy()


rmse = np.sqrt(mean_squared_error(y_test_np, test_predictions_np))
mae = mean_absolute_error(y_test_np, test_predictions_np)
r2 = r2_score(y_test_np, test_predictions_np)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)


# Additional function to filter data for 2017 season
def get_2017(data):
    temp = data.copy(deep=False)
    temp = temp[temp['NEXT_SEASON'] == 2018]
    return temp

# Get the 2017 data
data_2017 = get_2017(data_scaled)
X_2017, y_2018 = split_data_X_y(data_2017)

# Convert data to PyTorch tensors
X_2017_tensor = torch.tensor(X_2017.values, dtype=torch.float32)

# Predict using the trained model
model.eval()
with torch.no_grad():
    y_2018_pred_tensor = model(X_2017_tensor.unsqueeze(1)).flatten()

# Convert tensor to numpy array for further processing
y_2018_pred = y_2018_pred_tensor.numpy()

# Create a dictionary to map team IDs to team names
team_id_name = {row['TEAM_ID']: row['NICKNAME'] for _, row in teams.iterrows()}

# Generate the list of team names corresponding to the team IDs in the 2017 data
name_list = [team_id_name[team_id] for team_id in data_2017['TEAM_ID']]

# Create DataFrame to save the predictions and actual wins
prediction_dict_2018 = {
    "team_name": name_list,
    "wins_pred_2018": np.round(y_2018_pred),
    "wins_2018": y_2018['NEXT_W'].values
}
prediction_df_2018 = pd.DataFrame(prediction_dict_2018)
prediction_df_2018 = prediction_df_2018.astype({'wins_pred_2018': 'int64'})
prediction_df_2018.sort_values(by='wins_pred_2018', ascending=False, inplace=True)
prediction_df_2018.reset_index(inplace=True, drop=True)

# Calculate MAE and RMSE for 2018 predictions
mae_2018 = mean_absolute_error(prediction_df_2018['wins_2018'], prediction_df_2018['wins_pred_2018'])
rmse_2018 = np.sqrt(mean_squared_error(prediction_df_2018['wins_2018'], prediction_df_2018['wins_pred_2018']))

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

prediction_df_2018.to_csv('nba_data_2.csv', index=False)

print("done")
