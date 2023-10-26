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
data_train = data_scaled[data_scaled['NEXT_SEASON'] != 2018]
X, y = split_data_X_y(data_train)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=2023)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training, validation, and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize neural network, loss function, optimizer, and scheduler
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)

# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 3
counter = 0

# Training loop
for epoch in range(100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")

    # Step the scheduler
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        if counter >= patience:
            print("Early stopping")
            break

# Additional function to filter data for 2017 season
def get_2017(data):
    temp = data.copy(deep=False)
    temp = temp[temp['NEXT_SEASON'] == 2018]
    return temp

# Evaluation
model.eval()
with torch.no_grad():
    # Testing on X_test
    test_predictions = model(X_test_tensor).flatten()
    test_loss = criterion(test_predictions, y_test_tensor.flatten())
    print(f"Test Loss (MSE): {test_loss.item()}")

    # Metrics
    y_test_np = y_test_tensor.numpy().flatten()
    test_predictions_np = test_predictions.numpy()
    rmse = np.sqrt(mean_squared_error(y_test_np, test_predictions_np))
    mae = mean_absolute_error(y_test_np, test_predictions_np)
    r2 = r2_score(y_test_np, test_predictions_np)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)

    # Predictions for 2017 season
    data_2017 = get_2017(data_scaled)
    X_2017, y_2018 = split_data_X_y(data_2017)
    X_2017_tensor = torch.tensor(X_2017.values, dtype=torch.float32)
    y_2018_pred_tensor = model(X_2017_tensor).flatten()
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

# Calculate MAE, RMSE, and R² for 2018 predictions
mae_2018 = mean_absolute_error(prediction_df_2018['wins_2018'], prediction_df_2018['wins_pred_2018'])
rmse_2018 = np.sqrt(mean_squared_error(prediction_df_2018['wins_2018'], prediction_df_2018['wins_pred_2018']))
r2_2018 = r2_score(prediction_df_2018['wins_2018'], prediction_df_2018['wins_pred_2018'])

# Visualization
plt.figure(figsize=(14, 7))
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_2018'], color='blue', label='Actual Wins', s=100)
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_pred_2018'], color='red', label='Predicted Wins', s=100, alpha=0.6)
plt.xticks(rotation=90)
plt.ylabel('Number of Wins')
plt.title(f'Actual vs Predicted Wins for 2018 Season')
plt.legend()

# Add metrics to the plot
plt.annotate(f'MAE: {mae_2018:.2f}', xy=(0.75, 0.9), xycoords='axes fraction', fontsize=12)
plt.annotate(f'RMSE: {rmse_2018:.2f}', xy=(0.75, 0.85), xycoords='axes fraction', fontsize=12)
plt.annotate(f'R²: {r2_2018:.2f}', xy=(0.75, 0.8), xycoords='axes fraction', fontsize=12)

plt.tight_layout()
plt.show()


# Save to CSV
prediction_df_2018.to_csv('~/Desktop/nba-games/nba_data_2018_prediction.csv', index=False)
