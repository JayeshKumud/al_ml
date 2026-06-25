# HR Analytics – Employee Attrition Prediction (Logistic Regression)

This project analyzes HR data to understand why employees leave a company and builds a **Logistic Regression model** to predict employee attrition.  
The dataset is sourced from Kaggle:  
https://www.kaggle.com/giripujar/hr-analytics

---

## 📌 Project Objectives

1. Perform **Exploratory Data Analysis (EDA)** to understand key factors influencing employee attrition.
2. Identify the most important features that contribute to employees leaving.
3. Build a **Logistic Regression** model to predict whether an employee will leave.
4. Evaluate model performance using accuracy and predictions.

---

## 📊 Why Exploratory Data Analysis (EDA) Is Important

EDA is the foundation of any machine‑learning project.  
It helps you:

### **1. Understand the data**
Before modeling, we must know:
- What variables exist  
- How they behave  
- Whether they differ between employees who left vs stayed  

### **2. Identify meaningful features**
EDA revealed strong patterns:
- Low satisfaction → higher attrition  
- High monthly hours → higher attrition  
- No promotion → higher attrition  
- Low salary → higher attrition  

These insights directly guided **feature selection**.

### **3. Detect irrelevant or weak features**
Department showed only minor differences across attrition groups.  
Including weak features adds noise, so we excluded it.

### **4. Choose the right model**
Since the target variable is binary (0 = stay, 1 = leave),  
**Logistic Regression** is the correct model.

### **5. Interpret the model later**
EDA helps explain *why* the model predicts attrition.

---

## 📈 Key Insights from EDA

### **1. Satisfaction Level**
- Employees who left: **~0.44**
- Employees who stayed: **~0.66**

Lower satisfaction strongly correlates with leaving.

### **2. Average Monthly Hours**
- Left: **~207 hours**
- Stayed: **~199 hours**

Overworked employees are more likely to leave.

### **3. Promotion in Last 5 Years**
- Employees with promotions tend to stay.
- Lack of promotion increases attrition.

### **4. Salary Level**
Bar charts show:
- High salary → low attrition  
- Low salary → high attrition  

### **5. Department**
Some variation exists, but not significant enough to be a strong predictor.  
Thus, **department is excluded** from the model.

---

## 🧠 Features Selected for the Model

Based on EDA, the following features were chosen:

| Feature | Reason |
|--------|--------|
| **satisfaction_level** | Strong negative correlation with attrition |
| **average_montly_hours** | Overworking increases attrition |
| **promotion_last_5years** | Promotions improve retention |
| **salary** (dummy encoded) | Higher salary reduces attrition |

Salary is categorical, so it is converted into:
- salary_low  
- salary_medium  
- salary_high  

---

## 🤖 Model Used: Logistic Regression

Logistic Regression is ideal because:
- The target variable is binary (0 or 1)
- It outputs probabilities
- It is interpretable
- It works well with numeric and dummy‑encoded features

The model predicts:
- **0 → Employee stays**
- **1 → Employee leaves**

---

## 🧪 Model Training & Evaluation

The dataset is split into:
- **30% training**
- **70% testing**

After training:
- Predictions are generated using `model.predict(X_test)`
- Accuracy is calculated using `model.score(X_test, y_test)`

This gives a clear measure of how well the model generalizes.

---

## 📂 Project Structure (Recommended)