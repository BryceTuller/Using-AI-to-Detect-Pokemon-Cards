from preprocess import linear_pre_process
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------PRE PROCESS IMAGES-------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

x_is_pokemon, y_is_pokemon = linear_pre_process("data/PokemonCards", label=1)
x_isnt_pokemon, y_isnt_pokemon = linear_pre_process("data/NOTPokemonCards", label=0)

x = np.vstack((x_is_pokemon, x_isnt_pokemon))
y = np.concatenate((y_is_pokemon, y_isnt_pokemon))

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------TRAIN AND TEST MODEL-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_prediction_continuous = model.predict(x_test)
y_prediction_continuous = (y_prediction_continuous >= 0.8).astype(int)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------PRINT REPORT AND PLOTS-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

print("Classification Report : ")
print(classification_report(y_test, y_prediction_continuous))
print("Accuracy : " + str(accuracy_score(y_test, y_prediction_continuous)))

plt.figure(figsize=(8, 6))
plt.hist(y_prediction_continuous[y_test == 0], bins=20, alpha=0.5, label="Actual Class 0")
plt.hist(y_prediction_continuous[y_test == 1], bins=20, alpha=0.5, label="Actual Class 1")
plt.axvline(0.8, color='r', linestyle='--', label="Decision Boundary (0.8)")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Values for Test Set")
plt.legend()
plt.show()
