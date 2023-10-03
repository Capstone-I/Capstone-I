import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

games = pd.read_csv('~/Desktop/nba-games/games.csv')
details = pd.read_csv('~/Desktop/nba-games/games_details.csv')
teams = pd.read_csv('~/Desktop/nba-games/teams.csv')
ranking = pd.read_csv('~/Desktop/nba-games/ranking.csv')

def get_labels(ranking):
    # Deep copy of the dataframe
    temp = ranking.copy(deep=True)

    # Group by and get max values
    temp = temp.groupby(['TEAM_ID','SEASON_ID'])[['G','W']].max().reset_index()

    # Adjust 'SEASON_ID' values
    mask = temp['SEASON_ID'] / 10000 > 2
    temp.loc[mask, 'SEASON_ID'] = temp.loc[mask, 'SEASON_ID'] % 10000

    # Drop rows based on conditions
    drop_mask = (~mask) | (temp['G'] != 82)
    temp = temp[~drop_mask]

    # Drop unnecessary columns
    temp.drop(columns=['G'], inplace=True)

    return temp

def get_features(games, details):
    # Merge the dataframes on 'GAME_ID'
    temp = pd.merge(games, details, how='left', on='GAME_ID')
    
    # Keep specific columns and groupby to calculate sum
    columns_to_keep = ['TEAM_ID', 'SEASON', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 
                       'TO', 'PF', 'PTS', 'PLUS_MINUS']
    temp = temp[columns_to_keep]
    temp = temp.groupby(['TEAM_ID','SEASON']).sum().reset_index()

    # Calculate 'NEXT_SEASON' column
    temp['NEXT_SEASON'] = temp['SEASON'] + 1

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
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize neural network, loss function and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
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
y_2018_tensor = torch.tensor(y_2018.values, dtype=torch.float32)

# Predict using the trained model
model.eval()
with torch.no_grad():
    y_2018_pred_tensor = model(X_2017_tensor).flatten()

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

# Save to CSV
prediction_df_2018.to_csv('~/Desktop/nba-games/nba_data_2018_prediction.csv', index=False)