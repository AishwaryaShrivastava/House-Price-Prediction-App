# 🏡 House Price Prediction App

This is a web-based **House Price Prediction App** built using **Streamlit** and **Machine Learning**. The app predicts house and land prices based on various property features such as area, number of bedrooms, location type, and more.

---

## 📌 Features

✅ Predict house prices using a trained ML model  
✅ Clean and responsive UI with background images  
✅ Secure login/signup authentication using CSV  
✅ Dropdown menus for categorical features  
✅ Real-time price estimation  
✅ Background image and logo support  

---

## 💡 Technologies Used

- **Python**
- **Streamlit** – Frontend/UI
- **Scikit-learn** – Machine Learning
- **Pandas & NumPy** – Data Handling
- **Joblib** – Model Serialization
- **CSV** – User authentication data storage

---

## 🏗️ Project Structure

house_price_prediction_app/
├── app.py # Streamlit application file
├── train_model.py # Script to generate synthetic data and train ML model
├── users.csv # Stores registered users (username, password)
├── model/
│ └── house_price_model.pkl # Trained RandomForestRegressor model
├── assets/
│ ├── bg_image.jpg # Background image
│ ├── logo.png # Logo (optional)
│ └── house_sample.jpg # Display image inside the app



---

## ⚙️ How to Run

### 🔧 1. Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction-app.git
cd house-price-prediction-app
🐍 2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
🧠 3. Train the Model
bash
Copy
Edit
python train_model.py
🚀 4. Run the App
bash
Copy
Edit
streamlit run app.py
✍️ Usage
Open the app in your browser (usually http://localhost:8501)

Sign up or log in with your credentials

Fill in house-related input fields

Click "Predict Price"

View the estimated house price on the screen



📂 Sample User Credentials

Username: admin
Password: house123
📚 Project Purpose
This app is ideal for:

Real estate businesses looking for AI integration

ML learners building portfolio projects

Final-year students showcasing real-world applications

🔐 Security Note
Currently, user authentication is handled via plaintext in a CSV file (users.csv). For production use, implement proper encryption and switch to a database.



🙋‍♀️ Author
Aishwarya Shrivastava
📍 BIT Durg | AI Enthusiast | Python & ML Developer
📫 Contact: aishwaryashrivastava2004@gmail.com
