print("=" * 60)
print("STEP 1: Problem Definition & Setup")
print("=" * 60)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

PROBLEM = {
    "name": "Mobile Addiction Predictor",
    "task": "Multi-class Classification",
    "target": "addiction_level",
    "classes": ["No Addiction", "Mild", "Moderate", "Severe"],
    "class_codes": [0, 1, 2, 3],
    "primary_metric": "F1-Score (weighted)",
    "secondary": ["Accuracy", "Precision", "Recall"],
}

print("\nProblem Statement:")
for k, v in PROBLEM.items():
    print(f"  {k:<20}: {v}")

os.makedirs("outputs", exist_ok=True)

print("\n✓ Step 1 complete — outputs/ folder created")

print("\n" + "=" * 60)
print("STEP 2: Data Collection")
print("=" * 60)

def generate_dataset(n_samples=1000):

    screen_time = np.random.normal(5.5, 2.5, n_samples).clip(0, 16)
    pickups = np.random.normal(70, 40, n_samples).clip(0, 200)
    social_media = np.random.normal(2.5, 1.5, n_samples).clip(0, 12)
    gaming = np.random.normal(1.2, 1.0, n_samples).clip(0, 8)
    notifications = np.random.normal(90, 60, n_samples).clip(0, 300)
    sleep_hours = np.random.normal(6.8, 1.2, n_samples).clip(3, 12)
    offline_streak = np.random.normal(4.0, 3.0, n_samples).clip(0, 24)
    work_on_phone = np.random.normal(2.0, 1.5, n_samples).clip(0, 10)
    age = np.random.normal(27, 8, n_samples).clip(10, 70)

    meal_usage = np.random.binomial(1, 0.55, n_samples)
    sleep_usage = np.random.binomial(1, 0.60, n_samples)
    battery_anxiety = np.random.binomial(1, 0.45, n_samples)
    hidden_usage = np.random.binomial(1, 0.25, n_samples)

    raw = (
        screen_time / 16 * 0.22
        + pickups / 200 * 0.18
        + social_media / 12 * 0.16
        + work_on_phone / 10 * 0.10
        + gaming / 8 * 0.10
        + notifications / 300 * 0.09
        + (1 - sleep_hours / 12) * 0.08
        + (1 - offline_streak / 24) * 0.07
        + meal_usage * 0.045
        + sleep_usage * 0.055
        + battery_anxiety * 0.060
        + hidden_usage * 0.070
        + np.where(age < 25, 0.03, 0)
    )

    raw = raw + np.random.normal(0, 0.05, n_samples)
    raw = raw.clip(0, 1)

    labels = np.where(raw < 0.25, 0,
             np.where(raw < 0.45, 1,
             np.where(raw < 0.68, 2, 3)))

    df = pd.DataFrame({
        "age": age.round(0).astype(int),
        "screen_time": screen_time.round(1),
        "pickups": pickups.round(0).astype(int),
        "social_media": social_media.round(1),
        "gaming": gaming.round(1),
        "notifications": notifications.round(0).astype(int),
        "sleep_hours": sleep_hours.round(1),
        "offline_streak": offline_streak.round(1),
        "work_on_phone": work_on_phone.round(1),
        "meal_usage": meal_usage,
        "sleep_usage": sleep_usage,
        "battery_anxiety": battery_anxiety,
        "hidden_usage": hidden_usage,
        "addiction_level": labels,
    })

    return df


df = generate_dataset(1000)

df.to_csv("outputs/raw_dataset.csv", index=False)

print(f"\nDataset shape : {df.shape}")
print(f"Features      : {df.shape[1] - 1}")
print(f"Samples       : {df.shape[0]}")

print("\nFirst 5 rows:")
print(df.head())

print("\n✓ Step 2 complete — outputs/raw_dataset.csv saved")

print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop("addiction_level", axis=1)
y = df["addiction_level"]

FEATURE_BOUNDS = {
    "age": (10, 70),
    "screen_time": (0, 16),
    "pickups": (0, 200),
    "social_media": (0, 12),
    "gaming": (0, 8),
    "notifications": (0, 300),
    "sleep_hours": (3, 12),
    "offline_streak": (0, 24),
    "work_on_phone": (0, 10),
}

for col, (lo, hi) in FEATURE_BOUNDS.items():
    X[col] = X[col].clip(lo, hi)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = MinMaxScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

print("\n✓ Step 3 complete")

print("\n" + "=" * 60)
print("STEP 4: Feature Selection")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

rf_selector = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_selector.fit(X_train_scaled, y_train)

importances = pd.Series(
    rf_selector.feature_importances_,
    index=X_train.columns
)

importances = importances.sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

IMPORTANCE_THRESHOLD = 0.03

selected_features = importances[
    importances >= IMPORTANCE_THRESHOLD
].index.tolist()

print("\nSelected Features:")
print(selected_features)

X_train_sel = X_train_scaled[selected_features]
X_test_sel = X_test_scaled[selected_features]

print("\n✓ Step 4 complete")

print("\n" + "=" * 60)
print("STEP 5: Model Training")
print("=" * 60)

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nBaseline Model Scores:")

cv_results = {}

for name, model in baseline_models.items():

    scores = cross_val_score(
        model,
        X_train_sel,
        y_train,
        cv=cv,
        scoring="f1_weighted"
    )

    cv_results[name] = scores.mean()

    print(f"{name:<25}: {scores.mean():.4f}")

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [6, 8, None],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_search.fit(X_train_sel, y_train)

best_params = grid_search.best_params_

print("\nBest Parameters:")
print(best_params)

final_model = RandomForestClassifier(
    **best_params,
    random_state=42
)

final_model.fit(X_train_sel, y_train)

print("\n✓ Step 5 complete")

print("\n" + "=" * 60)
print("STEP 6: Prediction System")
print("=" * 60)

CLASS_NAMES = [
    "No Addiction",
    "Mild Addiction",
    "Moderate Addiction",
    "Severe Addiction"
]

CLASS_EMOJI = ["✅", "⚠️", "🔶", "🚨"]


def _generate_recommendations(inp, level):

    recs = []

    if inp.get("screen_time", 0) > 6:
        recs.append(
            "Limit daily screen time to 4–5 hrs."
        )

    if inp.get("pickups", 0) > 80:
        recs.append(
            "Enable Focus Mode to reduce pickups."
        )

    if inp.get("social_media", 0) > 3:
        recs.append(
            "Reduce social media usage."
        )

    if inp.get("sleep_usage", 0):
        recs.append(
            "Keep phone outside bedroom during sleep."
        )

    if inp.get("meal_usage", 0):
        recs.append(
            "Avoid phone usage during meals."
        )

    if inp.get("battery_anxiety", 0):
        recs.append(
            "Try reducing phone dependency gradually."
        )

    if inp.get("notifications", 0) > 100:
        recs.append(
            "Reduce unnecessary notifications."
        )

    if inp.get("offline_streak", 0) < 2:
        recs.append(
            "Schedule daily phone-free time."
        )

    if inp.get("sleep_hours", 7) < 6:
        recs.append(
            "Phone usage is affecting your sleep quality."
        )

    if inp.get("hidden_usage", 0):
        recs.append(
            "Consider discussing usage habits with someone trusted."
        )

    if level == 0:
        recs.append(
            "Excellent habits! Keep maintaining balance."
        )

    return recs if recs else [
        "Your usage looks balanced."
    ]


def predict_addiction(user_input):

    input_df = pd.DataFrame([user_input])

    for col, (lo, hi) in FEATURE_BOUNDS.items():

        if col in input_df.columns:
            input_df[col] = input_df[col].clip(lo, hi)

    input_scaled = pd.DataFrame(
        scaler.transform(input_df[X_train.columns]),
        columns=X_train.columns
    )

    input_final = input_scaled[selected_features]

    pred_class = final_model.predict(input_final)[0]

    pred_proba = final_model.predict_proba(input_final)[0]

    recs = _generate_recommendations(
        user_input,
        pred_class
    )

    return {
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "confidence": pred_proba[pred_class],
        "probabilities": dict(
            zip(CLASS_NAMES, pred_proba.round(4))
        ),
        "recommendations": recs
    }


test_user = {
    "age": 20,
    "screen_time": 9,
    "pickups": 120,
    "social_media": 5,
    "gaming": 2,
    "notifications": 180,
    "sleep_hours": 5,
    "offline_streak": 1,
    "work_on_phone": 3,
    "meal_usage": 1,
    "sleep_usage": 1,
    "battery_anxiety": 1,
    "hidden_usage": 0,
}

result = predict_addiction(test_user)

print("\nPrediction Result:")
print(f"Class       : {result['predicted_label']}")
print(f"Confidence  : {result['confidence']:.2%}")

print("\nRecommendations:")
for r in result["recommendations"]:
    print("-", r)

print("\n✓ Step 6 complete")

print("\n" + "=" * 60)
print("STEP 7: Evaluation")
print("=" * 60)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_pred = final_model.predict(X_test_sel)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["None", "Mild", "Moderate", "Severe"],
    yticklabels=["None", "Mild", "Moderate", "Severe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150
)

plt.close()

print("\n✓ Step 7 complete")

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)

print("\nFiles Saved:")
print("1. outputs/raw_dataset.csv")
print("2. outputs/confusion_matrix.png")

print("\nModel Successfully Built ✓")  

print("\n" + "=" * 60)
print("SIMPLE MOBILE ADDICTION CHECKER")
print("=" * 60)

# Simple user inputs
social_media = float(input("Enter social media hours per day: "))
chatting = float(input("Enter chatting/screen hours per day: "))
notifications = int(input("Enter daily notifications: "))

# Create simple user data
simple_user = {
    "age": 20,
    "screen_time": chatting,
    "pickups": notifications // 2,
    "social_media": social_media,
    "gaming": 1,
    "notifications": notifications,
    "sleep_hours": 7,
    "offline_streak": 3,
    "work_on_phone": 1,
    "meal_usage": 0,
    "sleep_usage": 0,
    "battery_anxiety": 0,
    "hidden_usage": 0,
}

# Predict result
result = predict_addiction(simple_user)

print("\n" + "=" * 40)
print("PREDICTION RESULT")
print("=" * 40)

print(f"Addiction Level : {result['predicted_label']}")
print(f"Confidence      : {result['confidence']:.2%}")

print("\nRecommendations:")
for rec in result["recommendations"]:
    print("-", rec)


    print("\n" + "=" * 60)
print("SIMPLE MOBILE ADDICTION CHECKER")
print("=" * 60)

# User Inputs
age = int(input("Enter your age: "))
screen_time = float(input("Enter total screen/chatting hours per day: "))
social_media = float(input("Enter social media hours per day: "))
gaming = float(input("Enter gaming hours per day: "))
notifications = int(input("Enter daily notifications: "))
sleep_hours = float(input("Enter sleep hours per day: "))
pickups = int(input("How many times you check phone daily: "))
offline_streak = float(input("Phone-free hours daily: "))

meal_usage = int(input("Use phone during meals? (1=Yes, 0=No): "))
sleep_usage = int(input("Use phone before sleep? (1=Yes, 0=No): "))
battery_anxiety = int(input("Feel anxiety when battery is low? (1=Yes, 0=No): "))
hidden_usage = int(input("Hide phone usage from others? (1=Yes, 0=No): "))

# Create input dictionary
user_data = {
    "age": age,
    "screen_time": screen_time,
    "pickups": pickups,
    "social_media": social_media,
    "gaming": gaming,
    "notifications": notifications,
    "sleep_hours": sleep_hours,
    "offline_streak": offline_streak,
    "work_on_phone": 2,
    "meal_usage": meal_usage,
    "sleep_usage": sleep_usage,
    "battery_anxiety": battery_anxiety,
    "hidden_usage": hidden_usage,
}

# Prediction
result = predict_addiction(user_data)

print("\n" + "=" * 50)
print("MOBILE ADDICTION RESULT")
print("=" * 50)

print(f"Addiction Level : {result['predicted_label']}")
print(f"Confidence      : {result['confidence']:.2%}")

print("\nClass Probabilities:")
for k, v in result["probabilities"].items():
    print(f"{k:<20}: {v:.2%}")

print("\nRecommendations:")
for rec in result["recommendations"]:
    print("-", rec)