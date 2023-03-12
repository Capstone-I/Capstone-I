import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compiles the model
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluates the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'test loss: {test_loss:.3f}, test accuracy: {test_acc:.3f}')
