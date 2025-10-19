# House Price Prediction System

A machine learning-powered web application that predicts real estate prices using multiple regression models. Built with Flask and scikit-learn, this system compares three different ML algorithms to provide accurate price estimates with confidence intervals.

## 🏠 Project Overview

This application trains and compares multiple machine learning models to predict house prices based on property characteristics including:

- **Property Features**: Bedrooms, bathrooms, area (sq ft), floors, year built
- **Location Data**: Neighborhood/location classification
- **Property Condition**: Overall condition rating
- **Amenities**: Garage availability

The system automatically selects the best-performing model and provides predictions with detailed performance metrics.

## 🚀 Features

- **Multi-Model Comparison**: Trains and evaluates 3 different algorithms:
  - Linear Regression (Ridge)
  - Random Forest Regressor
  - Gradient Boosting Regressor

- **Automatic Model Selection**: Chooses the best model based on R² score performance
- **Feature Engineering**: Creates 20+ derived features for improved accuracy
- **Data Preprocessing**: Handles outliers, scaling, and encoding automatically
- **Interactive Web Interface**: Clean, responsive UI for price predictions
- **Performance Visualization**: Generates model comparison charts
- **Confidence Intervals**: Provides prediction ranges based on model variance

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Data Visualization**: Matplotlib
- **Frontend**: HTML5, CSS3 (Glass morphism design)
- **Model Persistence**: Pickle serialization

## 📁 Project Structure

```
├── app.py              # Main Flask application
├── model.pkl           # Trained ML models (generated)
├── data.csv            # Training dataset
├── static/             # Static assets (charts, images)
└── templates/
    ├── index.html      # Input form interface
    └── result.html     # Prediction results page
```

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser to `http://localhost:5000`

## 📊 Model Performance

The system targets **85%+ R² accuracy** and automatically selects the best-performing model:

- **Linear Regression**: Baseline model for feature relationship analysis
- **Random Forest**: Handles non-linear patterns, typically best accuracy
- **Gradient Boosting**: Sequential error correction, balanced speed/accuracy

Performance metrics include:
- R² Score (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Cross-validation scores

## 🎯 Usage

1. **Input Property Details**: Fill in the form with property characteristics
2. **Get Prediction**: System uses the best-trained model for price estimation
3. **View Results**: See predicted price, model confidence, and comparison table
4. **Confidence Range**: Review the 95% confidence interval for the prediction

## 🔧 Advanced Features

- **Automated Outlier Detection**: Removes statistical anomalies from training data
- **Feature Engineering Pipeline**: Creates derived features like age, ratios, and interactions
- **Robust Scaling**: Handles different feature scales automatically
- **One-Hot Encoding**: Processes categorical variables (location, condition, garage)
- **Cross-Validation**: Ensures model generalization with 5-fold CV

## 📈 Model Training Process

1. **Data Loading**: Reads and validates CSV dataset
2. **Outlier Removal**: Statistical filtering using IQR method
3. **Feature Creation**: Generates 20+ engineered features
4. **Data Splitting**: 75% training, 25% testing split
5. **Model Training**: Parallel training of 3 algorithms
6. **Performance Evaluation**: R², MAE, RMSE, and CV scoring
7. **Model Selection**: Automatic best model identification
8. **Persistence**: Saves trained models for production use

## 💡 Business Applications

- **Real Estate Valuation**: Property appraisal and market analysis
- **Investment Analysis**: ROI calculations for property investments
- **Market Research**: Price trend analysis and forecasting
- **Mortgage Lending**: Risk assessment for loan applications

## 🔍 Technical Highlights

- **Scalable Architecture**: Modular design for easy model updates
- **Production Ready**: Error handling, logging, and model validation
- **Performance Optimized**: Efficient preprocessing and prediction pipelines
- **User-Friendly Interface**: Intuitive web forms with validation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

*Built with ❤️ using Python, Flask, and Machine Learning*