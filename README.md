# Insurance Premium Predictor

A Machine Learning web application that predicts insurance premiums based on customer information such as age, BMI, salary, smoking habits, and region. The application also calculates the estimated final return amount based on policy duration and interest rate.

Built using **Python**, **Streamlit**, and **Scikit-learn**.

---

## 🚀 Features

* Predict insurance premium instantly
* Interactive Streamlit web interface
* Machine Learning-based prediction system
* Calculates:

  * Predicted yearly premium
  * Final return amount
* User-friendly form inputs
* Real-time prediction results

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Faker**
* **Pickle**

---

## 📂 Project Structure

```bash
Insurance-Premium-Predictor/
│
├── app.py                 # Streamlit web app
├── train_model.py         # Model training script
├── insurance.csv          # Dataset
├── model_gb.pkl           # Trained Gradient Boosting model
├── requirements.txt       # Required dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset Information

The dataset contains:

* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region
* Salary
* Insurance Charges

The dataset is generated programmatically using the `Faker` library and random values.

---

## 🧠 Machine Learning Model

The project uses:

### ✅ Gradient Boosting Regressor

The target variable (`charges`) is transformed using logarithmic scaling for better prediction performance.

### Feature Engineering Includes:

* Label encoding
* One-hot encoding
* MinMax scaling

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Insurance-Premium-Predictor.git
cd Insurance-Premium-Predictor
```

---

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Application will start at:

```bash
http://localhost:8501
```

---

## 🏋️ Train the Model

To retrain the model:

```bash
python train_model.py
```

This will:

* Generate dataset
* Train model
* Save model as `model_gb.pkl`

---

## 📥 Input Parameters

Users provide:

* Age
* Gender
* Annual Salary
* BMI
* Smoking Status
* Number of Children
* Region
* Interest Rate
* Policy Duration

---

## 📤 Output

The app predicts:

✅ Insurance Premium
✅ Final Return Amount

---

## 📸 Screenshots

Add your project screenshots here.

Example:

```markdown
![Home Page](screenshots/home.png)
```

---

## 📌 Future Enhancements

* Deploy on Streamlit Cloud
* Add user authentication
* Store prediction history
* Add advanced analytics dashboard
* Improve model accuracy with larger datasets

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed by **Rakesh**

GitHub: [Rakesh0622 GitHub](https://github.com/Rakesh0622?utm_source=chatgpt.com)
