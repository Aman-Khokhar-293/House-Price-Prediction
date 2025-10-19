import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# File paths
DATA_PATH = 'data.csv'
MODEL_PATH = 'model.pkl'
STATIC_DIR = 'static'

LOCATIONS = []
CONDITIONS = []
GARAGE_OPTIONS = []

app = Flask(__name__)
os.makedirs(STATIC_DIR, exist_ok=True)


def format_inr(amount):
    # Format numbers in Indian style with commas
    n = int(round(float(amount)))
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return "₹" + ",".join(parts + [last3])


def load_data():
    # Read the CSV file
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Can't find {DATA_PATH}. Make sure the file exists.")
    
    df = pd.read_csv(DATA_PATH)
    # Clean up column names - make them lowercase
    df.columns = ['bedrooms', 'bathrooms', 'area', 'floors', 
                  'yearbuilt', 'location', 'condition', 'garage', 'price']
    
    return df


def create_features(df):
    # Use only original CSV features - no feature engineering
    return df.copy()


def remove_outliers(df):
    # Get rid of weird data points
    df_clean = df.copy()
    n_before = len(df_clean)
    
    # Remove price outliers
    Q1 = df_clean['price'].quantile(0.10)
    Q3 = df_clean['price'].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 2.5 * IQR
    upper = Q3 + 2.5 * IQR
    df_clean = df_clean[(df_clean['price'] >= lower) & (df_clean['price'] <= upper)]
    
    # Area outliers
    Q1_area = df_clean['area'].quantile(0.05)
    Q3_area = df_clean['area'].quantile(0.95)
    IQR_area = Q3_area - Q1_area
    df_clean = df_clean[(df_clean['area'] >= Q1_area - 2.0 * IQR_area) & 
                        (df_clean['area'] <= Q3_area + 2.0 * IQR_area)]
    
    # Basic sanity checks
    df_clean = df_clean[df_clean['price'] > 0]
    df_clean = df_clean[df_clean['area'] > 0]
    df_clean = df_clean[df_clean['bedrooms'] >= 1]
    df_clean = df_clean[df_clean['bathrooms'] >= 1]
    
    n_after = len(df_clean)
    print(f"Removed {n_before - n_after} outliers (kept {n_after} rows)")
    
    return df_clean


def setup_preprocessor():
    # Setup the preprocessing steps - only basic CSV features
    numeric_cols = ['bedrooms', 'bathrooms', 'area', 'floors', 'yearbuilt']
    categorical_cols = ['location', 'condition', 'garage']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_cols),
        ]
    )
    
    return preprocessor, numeric_cols, categorical_cols


def make_plots(r2_dict, mae_dict, save_path):
    # Create some charts to visualize performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    model_names = list(r2_dict.keys())
    r2_values = [r2_dict[k] for k in model_names]
    mae_values = [mae_dict[k] for k in model_names]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # R2 chart
    bars1 = ax1.bar(model_names, r2_values, color=colors[:len(model_names)], 
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance - R² Score\n(Target: 85%+)', 
                  fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.axhline(y=0.85, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Target: 85%')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend()
    
    for bar, score in zip(bars1, r2_values):
        height = score
        color = 'green' if score >= 0.85 else 'black'
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f"{score:.1%}", ha='center', fontsize=10, 
                fontweight='bold', color=color)
    
    # MAE chart
    bars2 = ax2.bar(model_names, mae_values, color=colors[:len(model_names)], 
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Mean Absolute Error (₹)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance - MAE\n(Lower is Better)', 
                  fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, score in zip(bars2, mae_values):
        height = score
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(mae_values)*0.03, 
                f"₹{score/1000:.0f}K", ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def set_dropdown_options(df):
    # Set the options for dropdowns in the UI
    global LOCATIONS, CONDITIONS, GARAGE_OPTIONS
    LOCATIONS = sorted(df['location'].unique().tolist())
    CONDITIONS = sorted(df['condition'].unique().tolist())
    GARAGE_OPTIONS = sorted(df['garage'].unique().tolist())


# ==================== MODEL 1: LINEAR REGRESSION ====================
def train_linear_model(X_train, y_train, X_test, y_test, preprocessor):
    """
    Linear Regression Model (Ridge)
    - Simple baseline model
    - Good for understanding feature relationships
    - Fast training and prediction
    """
    print("\n[1/3] Training Linear Regression Model...")
    
    model = Ridge(alpha=15.0)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    
    status = "PASS" if test_r2 >= 0.85 else "FAIL"
    print(f"  [{status}] Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"        Train R²: {train_r2:.4f}")
    print(f"        CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"        MAE: ₹{mae:,.0f}")
    print(f"        RMSE: ₹{rmse:,.0f}")
    
    return pipeline, test_r2, mae


# ==================== MODEL 2: RANDOM FOREST ====================
def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    """
    Random Forest Model
    - Ensemble of decision trees
    - Handles non-linear relationships well
    - Generally gives best accuracy
    """
    print("\n[2/3] Training Random Forest Model...")
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    
    status = "PASS" if test_r2 >= 0.85 else "FAIL"
    print(f"  [{status}] Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"        Train R²: {train_r2:.4f}")
    print(f"        CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"        MAE: ₹{mae:,.0f}")
    print(f"        RMSE: ₹{rmse:,.0f}")
    
    return pipeline, test_r2, mae


# ==================== MODEL 3: GRADIENT BOOSTING ====================
def train_gradient_boosting(X_train, y_train, X_test, y_test, preprocessor):
    """
    Gradient Boosting Model
    - Sequential ensemble method
    - Corrects errors from previous trees
    - Good balance of speed and accuracy
    """
    print("\n[3/3] Training Gradient Boosting Model...")
    
    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=4,
        subsample=0.9,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    
    status = "PASS" if test_r2 >= 0.85 else "FAIL"
    print(f"  [{status}] Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"        Train R²: {train_r2:.4f}")
    print(f"        CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"        MAE: ₹{mae:,.0f}")
    print(f"        RMSE: ₹{rmse:,.0f}")
    
    return pipeline, test_r2, mae


def train_models():
    print("\n" + "="*70)
    print("Starting model training...")
    print("="*70)
    
    # Load and prep the data
    df_raw = load_data()
    print(f"\nLoaded {len(df_raw)} records from CSV")
    
    df_clean = remove_outliers(df_raw)
    df_final = create_features(df_clean)
    
    set_dropdown_options(df_clean)
    print(f"Found locations: {LOCATIONS}")
    print(f"Found conditions: {CONDITIONS}")
    print(f"Found garage options: {GARAGE_OPTIONS}")
    
    # Define what features to use - only original CSV features
    all_features = [
        'bedrooms', 'bathrooms', 'area', 'floors', 'yearbuilt',
        'location', 'condition', 'garage'
    ]
    
    X = df_final[all_features]
    y = df_final['price']
    
    # Split data into train and test sets (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"\nSplit data: {len(X_train)} train, {len(X_test)} test")
    
    preprocessor, num_feats, cat_feats = setup_preprocessor()
    
    print("\n" + "="*70)
    print("Training 3 different models...")
    print("="*70)
    
    # Train each model separately
    linear_pipeline, linear_r2, linear_mae = train_linear_model(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    rf_pipeline, rf_r2, rf_mae = train_random_forest(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    gb_pipeline, gb_r2, gb_mae = train_gradient_boosting(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    # Store results
    trained_models = {
        'LinearRegression': linear_pipeline,
        'RandomForest': rf_pipeline,
        'GradientBoosting': gb_pipeline
    }
    
    results_r2 = {
        'LinearRegression': float(linear_r2),
        'RandomForest': float(rf_r2),
        'GradientBoosting': float(gb_r2)
    }
    
    results_mae = {
        'LinearRegression': float(linear_mae),
        'RandomForest': float(rf_mae),
        'GradientBoosting': float(gb_mae)
    }
    
    # Pick the best one
    best_model_name = max(results_r2, key=results_r2.get)
    best_score = results_r2[best_model_name]
    
    print("\n" + "="*70)
    print(f"Best model: {best_model_name}")
    print(f"R² Score: {best_score:.4f} ({best_score*100:.2f}%)")
    
    if best_score >= 0.85:
        print("SUCCESS! Hit the 85% target")
    else:
        print(f"Not quite there yet. Got {best_score*100:.2f}% (need 85%+)")
    
    print("="*70 + "\n")
    
    # Save everything
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'pipelines': trained_models,
            'best_name': best_model_name,
            'r2_scores': results_r2,
            'mae_scores': results_mae,
            'feature_schema': list(all_features)
        }, f)
    
    print(f"Saved model to {MODEL_PATH}")
    
    # Make the charts
    plot_path = os.path.join(STATIC_DIR, 'model_r2.png')
    make_plots(results_r2, results_mae, plot_path)
    print(f"Saved charts to {plot_path}\n")
    
    return trained_models, best_model_name, results_r2, results_mae


def load_saved_model():
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                pipelines = data.get('pipelines')
                best_name = data.get('best_name', 'RandomForest')
                r2_scores = data.get('r2_scores', {})
                mae_scores = data.get('mae_scores', {})
                
                # Load the dropdown options
                try:
                    df = load_data()
                    set_dropdown_options(df)
                except Exception:
                    pass
                
                if pipelines is None:
                    print("Model file is incomplete, need to retrain")
                    return train_models()
                
                return pipelines, best_name, r2_scores, mae_scores
        except Exception as e:
            print(f"Couldn't load model ({e}), training new one")
    
    return train_models()


# Load model when app starts
PIPELINES, BEST_MODEL_NAME, R2_SCORES, MAE_SCORES = load_saved_model()
print(f"\n{'='*70}")
print(f"Using model: {BEST_MODEL_NAME}")
print(f"R² Score: {R2_SCORES.get(BEST_MODEL_NAME, 0):.4f} ({R2_SCORES.get(BEST_MODEL_NAME, 0)*100:.2f}%)")
print(f"{'='*70}\n")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                         locations=LOCATIONS, 
                         conditions=CONDITIONS, 
                         garage_options=GARAGE_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form values
        bedrooms = int(request.form.get('bedrooms', 2))
        bathrooms = int(request.form.get('bathrooms', 2))
        area = float(request.form.get('area', 2000))
        floors = int(request.form.get('floors', 2))
        yearbuilt = int(request.form.get('yearbuilt', 2000))
        location = request.form.get('location', LOCATIONS[0] if LOCATIONS else 'Downtown')
        condition = request.form.get('condition', CONDITIONS[0] if CONDITIONS else 'Good')
        garage = request.form.get('garage', GARAGE_OPTIONS[0] if GARAGE_OPTIONS else 'No')
        
        # Make a dataframe from the input
        input_data = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'area': area,
            'floors': floors,
            'yearbuilt': yearbuilt,
            'location': location,
            'condition': condition,
            'garage': garage
        }])
        
        # Add the same features we used in training
        input_data = create_features(input_data)
        
        # Get predictions from all models
        all_predictions = []
        print(f"DEBUG: R2_SCORES = {R2_SCORES}")  # Debug output
        for name, pipeline in PIPELINES.items():
            try:
                pred = float(pipeline.predict(input_data)[0])
                price_formatted = format_inr(pred)
            except Exception:
                pred = float('nan')
                price_formatted = 'N/A'
            
            r2 = R2_SCORES.get(name, 0.0)
            mae = MAE_SCORES.get(name, 0.0)
            print(f"DEBUG: {name} - r2={r2}, r2_display={max(0, r2)}")  # Debug output
            
            all_predictions.append({
                'name': name,
                'price': pred,
                'price_str': price_formatted,
                'r2': r2,
                'r2_display': max(0, r2),
                'mae': mae
            })
        
        # Use the best model's prediction
        best_name = max(R2_SCORES.keys(), key=lambda x: R2_SCORES[x]) if R2_SCORES else 'RandomForest'
        best_prediction = next((p['price'] for p in all_predictions if p['name'] == best_name), 0)
        final_price = format_inr(best_prediction) if np.isfinite(best_prediction) else 'N/A'
        best_r2 = max(0, R2_SCORES.get(best_name, 0))
        
        # Calculate a confidence range
        valid_preds = [p['price'] for p in all_predictions if np.isfinite(p['price'])]
        if len(valid_preds) > 1:
            std_dev = np.std(valid_preds)
            lower_bound = format_inr(best_prediction - 1.96 * std_dev)
            upper_bound = format_inr(best_prediction + 1.96 * std_dev)
            conf_range = f"{lower_bound} - {upper_bound}"
        else:
            conf_range = "N/A"
        
        return render_template(
            'result.html',
            predicted_price=final_price,
            model_name=best_name,
            r2_score_display=f"{best_r2:.3f}",
            model_outputs=all_predictions,
            confidence_range=conf_range
        )
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template(
            'result.html', 
            predicted_price=f"Error: {e}", 
            model_name=BEST_MODEL_NAME, 
            r2_score_display="-", 
            model_outputs=[],
            confidence_range="N/A"
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
