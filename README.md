# 📱 Mobile Addiction Predictor

A Machine Learning based project that predicts a user's **mobile phone addiction level** using daily smartphone usage habits such as screen time, social media usage, notifications, sleep behavior, gaming hours, and phone-checking frequency.

The project uses multiple classification algorithms and predicts addiction levels into four categories:

* ✅ No Addiction
* ⚠️ Mild Addiction
* 🔶 Moderate Addiction
* 🚨 Severe Addiction

---

# 🚀 Project Features

* Synthetic dataset generation using Python
* Data preprocessing and feature scaling
* Feature importance analysis
* Multiple ML model comparison
* Hyperparameter tuning using GridSearchCV
* Mobile addiction prediction system
* Personalized recommendations
* Confusion matrix visualization
* CSV and image output generation

---

# 🧠 Machine Learning Models Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gradient Boosting
* Random Forest Classifier

---

# 📂 Project Structure

```text id="fp9p89"
Mobile-Addiction-Predictor/
│
├── outputs/
│   ├── raw_dataset.csv
│   └── confusion_matrix.png
│
├── mobile_addiction_predictor.py
├── README.md
└── requirements.txt
```

---

# ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# 📊 Dataset Features

The dataset contains behavioral smartphone usage features:

| Feature         | Description                 |
| --------------- | --------------------------- |
| age             | User age                    |
| screen_time     | Total screen time per day   |
| pickups         | Number of phone checks      |
| social_media    | Social media usage hours    |
| gaming          | Gaming hours                |
| notifications   | Daily notifications         |
| sleep_hours     | Daily sleep duration        |
| offline_streak  | Phone-free hours            |
| meal_usage      | Phone usage during meals    |
| sleep_usage     | Phone usage before sleep    |
| battery_anxiety | Anxiety when battery is low |
| hidden_usage    | Hiding phone usage          |

---

# 🎯 Addiction Classes

| Class Code | Addiction Level    |
| ---------- | ------------------ |
| 0          | No Addiction       |
| 1          | Mild Addiction     |
| 2          | Moderate Addiction |
| 3          | Severe Addiction   |

---

# 🔄 Project Workflow

1. Problem Definition
2. Data Collection
3. Data Preprocessing
4. Feature Selection
5. Model Training
6. Prediction System
7. Model Evaluation

---

# 📈 Evaluation Metrics

The project evaluates model performance using:

* Accuracy
* Precision
* Recall
* F1-Score

---

# 🖥️ Example User Input

```python id="4u2x76"
Enter your age: 20
Enter total screen/chatting hours per day: 8
Enter social media hours per day: 4
Enter gaming hours per day: 2
Enter daily notifications: 45
How many times you check phone daily: 90
Enter sleep hours per day: 5
Phone-free hours daily: 1
Use phone during meals? (1=Yes, 0=No): 1
Use phone before sleep? (1=Yes, 0=No): 1
Feel anxiety when battery is low? (1=Yes, 0=No): 1
Hide phone usage from others? (1=Yes, 0=No): 0
```

---

# 📌 Example Prediction Output

```text id="2stq3d"
Addiction Level : Moderate Addiction
Confidence      : 84.32%
```

---

# 💡 Recommendation System

The system also provides personalized suggestions such as:

* Reduce screen time
* Avoid phone usage during meals
* Improve sleep habits
* Reduce notifications
* Schedule phone-free time

---

# 📉 Confusion Matrix

The trained model automatically generates and saves a confusion matrix visualization:

```text id="0nl4hz"
outputs/confusion_matrix.png
```

---

# ▶️ How to Run the Project

## 1️⃣ Clone Repository

```bash id="v9d4j2"
git clone https://github.com/your-username/mobile-addiction-predictor.git
```

## 2️⃣ Open Project Folder

```bash id="h6ez9n"
cd mobile-addiction-predictor
```

## 3️⃣ Install Dependencies

```bash id="brm7qh"
pip install -r requirements.txt
```

## 4️⃣ Run Project

```bash id="fq2h11"
python mobile_addiction_predictor.py
```

---

# 📦 requirements.txt

```text id="m1nqzz"
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

# 🔮 Future Improvements

* Real-world dataset integration
* Deep Learning implementation
* Mobile App integration
* Web-based prediction dashboard
* Real-time screen tracking system

---

# 👨‍💻 Author

Developed by Tooba Iqbal
BS Artificial Intelligence Student


