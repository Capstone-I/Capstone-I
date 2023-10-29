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
from qiskit.circuit.library import ZFeatureMap, EfficientSU2, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
import numpy as np
from qiskit.algorithms.optimizers import COBYLA, POWELL, SPSA, L_BFGS_B
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-asu/main/pi-deluca',
)

options = Options()
options.execution.shots = 128
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

# Flatten y_train and y_test
y_train_array = y_train.to_numpy().ravel()
y_test_array = y_test.to_numpy().ravel()

num_qubits = 19

# Define the feature map and ansatz
feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)

# Quantum circuit
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)


#Set up the sampler qnn
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters, 
    estimator=estimator
)

# Set up the neural network classifier
regressor = NeuralNetworkRegressor(
    qnn,
    loss='squared_error',
    optimizer=COBYLA(maxiter=100)
)


# Convert to NumPy arrays
X_train = np.array(X_train)
y_train_array = y_train.to_numpy()

# Train the classifier
regressor.fit(X_train, y_train_array)

# Predict on test data
y_test_pred = regressor.predict(X_test)  
y_test_pred = y_test_pred.reshape(-1) 

# Compute metrics
r2 = regressor.score(y_test_array, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_array, y_test_pred))
mae = mean_absolute_error(y_test_array, y_test_pred)

# Print metrics
print("R2 Score:", r2)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Additional function to filter data for 2017 season
def get_2017(data):
    temp = data.copy(deep=False)
    temp = temp[temp['NEXT_SEASON'] == 2018]
    return temp

# Get the 2017 data
data_2017 = get_2017(data_scaled)
X_2017, y_2018 = split_data_X_y(data_2017)

# Predict using the trained QNN model
y_2018_pred = regressor.predict(X_2017)
y_2018_pred = y_2018_pred.reshape(-1)

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

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_2018'], color='blue', label='Actual Wins', s=100)
plt.scatter(prediction_df_2018['team_name'], prediction_df_2018['wins_pred_2018'], color='red', label='Predicted Wins', s=100, alpha=0.6)
plt.xticks(rotation=90)
plt.ylabel('Number of Wins')
plt.title(f'Actual vs Predicted Wins for 2018 Season\nMAE: {mae_2018:.2f} | RMSE: {rmse_2018:.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# Save to CSV
prediction_df_2018.to_csv('~/Desktop/nba-games/nba_data_2018_prediction_QNN.csv', index=False)