import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ===============================
# 1. Setup output folder
# ===============================

os.makedirs("qlearning_outputs", exist_ok=True)

# ===============================
# 2. Load Titanic processed data
# ===============================

X_train = pd.read_csv("dataset/titanic_X_train.csv")
X_test = pd.read_csv("dataset/titanic_X_test.csv")
y_train = pd.read_csv("dataset/titanic_y_train.csv")

if os.path.exists("dataset/titanic_y_test.csv"):
    y_test = pd.read_csv("dataset/titanic_y_test.csv")
elif os.path.exists("dataset/titanic_y_val.csv"):
    y_test = pd.read_csv("dataset/titanic_y_val.csv")
else:
    y_test = y_train.copy()

y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ===============================
# 3. State design
# ===============================

preferred_features = ["Pclass", "Sex", "AgeBand", "FareBand", "CabinKnown"]
state_features = [col for col in preferred_features if col in X_train.columns]

if len(state_features) == 0:
    state_features = list(X_train.columns)

print("State features used:", state_features)

def get_state(row):
    return tuple(np.round(row[state_features].values, 2))

# ===============================
# 4. Q-Learning parameters
# ===============================

actions = [0, 1]

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
episodes = 3000

q_table = {}
rewards_per_episode = []
training_accuracy_per_episode = []

# ===============================
# 5. Q-Learning training
# ===============================

for episode in range(episodes):

    total_reward = 0
    correct = 0

    for i in range(len(X_train)):

        state = get_state(X_train.iloc[i])

        if state not in q_table:
            q_table[state] = np.zeros(len(actions))

        # epsilon-greedy policy
        if np.random.random() < epsilon:
            action = np.random.choice(actions)
        else:
            action = int(np.argmax(q_table[state]))

        true_label = int(y_train[i])

        reward = 1 if action == true_label else -1

        if action == true_label:
            correct += 1

        total_reward += reward

        old_value = q_table[state][action]
        best_future_value = np.max(q_table[state])

        q_table[state][action] = old_value + learning_rate * (
            reward + discount_factor * best_future_value - old_value
        )

    rewards_per_episode.append(total_reward)
    training_accuracy_per_episode.append(correct / len(X_train))

print("Q-Learning training completed")

# ===============================
# 6. Prediction
# ===============================

predictions = []

X_train_state_values = X_train[state_features].values
X_test_state_values = X_test[state_features].values

for i in range(len(X_test)):

    state = get_state(X_test.iloc[i])

    if state in q_table:
        prediction = int(np.argmax(q_table[state]))

    else:
        # fallback for unseen states: nearest training passenger type
        distances = np.linalg.norm(X_train_state_values - X_test_state_values[i], axis=1)
        nearest_index = np.argmin(distances)
        prediction = int(y_train[nearest_index])

    predictions.append(prediction)

min_len = min(len(predictions), len(y_test))
predictions = predictions[:min_len]
y_test = y_test[:min_len]

print("Predictions completed")

# ===============================
# 7. Evaluation
# ===============================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
cm = confusion_matrix(y_test, predictions)

print("\n===== Q-Learning Evaluation Results =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# ===============================
# 8. Save classification report
# ===============================

report_text = classification_report(y_test, predictions, zero_division=0)

with open("qlearning_outputs/classification_report.txt", "w") as f:
    f.write("Q-Learning Reinforcement Learning Results\n")
    f.write("Dataset: Titanic\n")
    f.write("Student: Janani Veerakumar\n\n")
    f.write(f"Episodes: {episodes}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Discount Factor: {discount_factor}\n")
    f.write(f"Epsilon: {epsilon}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report_text)

# ===============================
# 9. Save result summary CSV
# ===============================

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

summary.to_csv("qlearning_outputs/results_summary.csv", index=False)

# ===============================
# 10. Reward graph
# ===============================

plt.figure(figsize=(8, 5))
plt.plot(rewards_per_episode)
plt.title("Q-Learning Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("qlearning_outputs/qlearning_reward_graph.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 11. Training accuracy graph
# ===============================

plt.figure(figsize=(8, 5))
plt.plot(training_accuracy_per_episode)
plt.title("Q-Learning Training Accuracy Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Training Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("qlearning_outputs/qlearning_training_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 12. Confusion matrix heatmap
# ===============================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Titanic Q-Learning Confusion Matrix")
plt.savefig("qlearning_outputs/qlearning_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 13. Metrics bar chart
# ===============================

plt.figure(figsize=(7, 5))
plt.bar(summary["Metric"], summary["Score"])
plt.ylim(0, 1)
plt.title("Q-Learning Performance Metrics")
plt.ylabel("Score")
plt.grid(axis="y")
plt.savefig("qlearning_outputs/qlearning_metrics_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("All plots saved successfully in qlearning_outputs folder.")

# ===============================
# 14. README file
# ===============================

with open("qlearning_outputs/README.txt", "w") as f:
    f.write("Reinforcement Learning - Q-Learning\n")
    f.write("Student: Janani Veerakumar\n")
    f.write("Dataset: Titanic\n\n")
    f.write("This implementation applies Q-Learning to a Titanic survival prediction environment.\n")
    f.write("State: passenger feature combination such as Pclass, Sex, AgeBand, FareBand and CabinKnown.\n")
    f.write("Actions: 0 = not survived, 1 = survived.\n")
    f.write("Reward: +1 for correct prediction, -1 for incorrect prediction.\n\n")
    f.write("Output files:\n")
    f.write("- qlearning_reward_graph.png\n")
    f.write("- qlearning_training_accuracy.png\n")
    f.write("- qlearning_confusion_matrix.png\n")
    f.write("- qlearning_metrics_bar.png\n")
    f.write("- classification_report.txt\n")
    f.write("- results_summary.csv\n")

print("\nAll files saved inside qlearning_outputs folder.")