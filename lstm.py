import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# 1. Load & Prepare Data (Strictly Chronological)
# ---------------------------------------------------------
# Load your dataset
df = pd.read_csv('sensor_data_rows.csv')

# Ensure dataset is sorted by timestamp to eliminate temporal data leakage
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').reset_index(drop=True)

# Select input features and target column
feature_cols = ['rain_percent', 'wind_speed_kmh', 'tof_distance_mm', 'om_temp', 'om_humidity']
target_col = 'status'  # Labels: 'Safe (LOW)', 'Warning (MODERATE)', 'Flood Alert (HIGH)'

# Scale features between 0 and 1
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols].fillna(0))

# Encode string target labels to integers (0, 1, 2)
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(df[target_col])
num_classes = len(np.unique(encoded_target))

# ---------------------------------------------------------
# 2. Create Time-Series Sliding Window Sequences
# ---------------------------------------------------------
TIME_STEPS = 10  # Look back at past 10 timesteps to predict the next state

def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(scaled_features, encoded_target, TIME_STEPS)

# ---------------------------------------------------------
# 3. Chronological Train-Test Split (NO Random Shuffling)
# ---------------------------------------------------------
# Split past 80% for training and latest 20% for testing
train_size = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# 4. Calculate Balanced Class Weights
# ---------------------------------------------------------
# Assigns higher loss penalties to rare classes (Safe / Warning)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weight_dict)

# ---------------------------------------------------------
# 5. Build LSTM Network
# ---------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------------
# 6. Train Model
# ---------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=60,
    batch_size=32,
    class_weight=class_weight_dict,  # Forces model to give equal weight to minority classes
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------
# 7. Evaluate and Plot Confusion Matrix
# ---------------------------------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n================== REVISED EVALUATION METRICS ==================")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_, 
    zero_division=0
))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('LSTM Confusion Matrix (Chronological Test Set)')
plt.show()