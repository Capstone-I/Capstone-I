import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from qiskit_ibm_runtime import Sampler, Options, QiskitRuntimeService, Estimator
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZFeatureMap, EfficientSU2, RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
import numpy as np
from qiskit.algorithms.optimizers import COBYLA, POWELL, SPSA, L_BFGS_B
from qiskit import QuantumCircuit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from qiskit_machine_learning.connectors import TorchConnector


service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-asu/main/pi-deluca',
)

options = Options()
options.execution.shots = 256
estimator = Estimator(backend='ibmq_qasm_simulator', options=options)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Define and create QNN
def create_qnn():
    feature_map = ZZFeatureMap(feature_dimension=8, reps=1)
    ansatz = EfficientSU2(num_qubits=8, reps=1) 
    qc = QuantumCircuit(8) 
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn

qnn = create_qnn()

class Net(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.qnn = TorchConnector(qnn)
        self.fc6 = nn.Linear(1, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))  
        x = self.qnn(x)
        x = self.fc6(x)  
        return x



# Initialize neural network, loss function and optimizer
model = Net(qnn)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()

# Training loop
for epoch in range(50):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

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

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_2018'], color='blue', label='Actual Wins', s=100)
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_pred_2018'], color='red', label='Predicted Wins', s=100, alpha=0.6)
plt.xticks(rotation=90)
plt.ylabel('Number of Wins')
plt.title(f'Actual vs Predicted Wins for 2018 Season\nMAE: {mae:.2f} | RMSE: {rmse:.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# Save to CSV
prediction_df_2018.to_csv('~/Desktop/nba-games/nba_data_2018_prediction_QNN.csv', index=False)