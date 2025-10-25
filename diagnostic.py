import pickle
import pandas as pd
import numpy as np
import os

# Load model and config
BASE_DIR = os.path.expanduser("~/Desktop/HIV_Prediction_Tool")
MODEL_PATH = os.path.join(BASE_DIR, 'final_best_model.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.pkl')

print("="*70)
print("🔍 MODEL DIAGNOSTIC REPORT")
print("="*70)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

print("\n1️⃣ MODEL INFORMATION")
print("-"*70)
print(f"Model Type: {type(model).__name__}")
print(f"Model Algorithm: {config['model_name']}")
print(f"Number of Features: {len(config['features'])}")

print("\n2️⃣ MODEL COEFFICIENTS/WEIGHTS")
print("-"*70)
if hasattr(model, 'coef_'):
    print("Model Coefficients:")
    coef_df = pd.DataFrame({
        'Feature': config['features'],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.head(10))
    
    # Check if all coefficients are similar (indicating problem)
    coef_std = np.std(model.coef_[0])
    print(f"\nCoefficient Standard Deviation: {coef_std:.4f}")
    if coef_std < 0.01:
        print("⚠️ WARNING: Very low coefficient variation - model may not be learning properly!")

if hasattr(model, 'intercept_'):
    print(f"\nIntercept: {model.intercept_}")

print("\n3️⃣ TRAINING METRICS")
print("-"*70)
for metric, value in config['metrics'].items():
    print(f"{metric:20s}: {value:.4f}")

print("\n4️⃣ CLASS DISTRIBUTION (from training)")
print("-"*70)
if 'data_preservation' in config:
    print(f"Total training records: {config['data_preservation']['preserved_rows']:,}")
    print(f"Preservation rate: {config['data_preservation']['preservation_rate']}")

print("\n5️⃣ TEST PREDICTIONS WITH DIFFERENT SCENARIOS")
print("-"*70)

# Create test cases
test_scenarios = {
    "Low Risk Patient": {
        'age': 30, 'cd4': 600, 'who stage': 0, 'total visits': 50,
        'days since last appointment': 20, 'lost to follow status': 0
    },
    "Medium Risk Patient": {
        'age': 45, 'cd4': 350, 'who stage': 1, 'total visits': 20,
        'days since last appointment': 45, 'lost to follow status': 0
    },
    "High Risk Patient": {
        'age': 50, 'cd4': 150, 'who stage': 3, 'total visits': 5,
        'days since last appointment': 90, 'lost to follow status': 1
    }
}

for scenario_name, scenario_data in test_scenarios.items():
    # Create full feature vector
    input_dict = {feat: 0 for feat in config['features']}
    input_dict.update(scenario_data)
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[config['features']]
    
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    print(f"\n{scenario_name}:")
    print(f"  Prediction: {'Non-Suppressed' if prediction == 1 else 'Suppressed'}")
    print(f"  Probabilities: Suppressed={proba[0]:.2%}, Non-Suppressed={proba[1]:.2%}")

print("\n6️⃣ CHECKING FOR ISSUES")
print("-"*70)

issues_found = []

# Check 1: Model always predicts same class
test_predictions = []
for _ in range(10):
    random_input = {feat: np.random.randint(0, 100) for feat in config['features']}
    input_df = pd.DataFrame([random_input])
    pred = model.predict(input_df)[0]
    test_predictions.append(pred)

unique_predictions = len(set(test_predictions))
if unique_predictions == 1:
    issues_found.append("❌ Model always predicts the SAME class (model bias!)")
else:
    print("✅ Model can predict different classes")

# Check 2: Low accuracy
if config['metrics']['accuracy'] < 0.6:
    issues_found.append(f"❌ Low accuracy ({config['metrics']['accuracy']:.2%}) - model not learning well")

# Check 3: High class imbalance
if config['metrics']['sensitivity'] < 0.5 or config['metrics']['specificity'] < 0.5:
    issues_found.append(f"❌ Poor sensitivity ({config['metrics']['sensitivity']:.2%}) or specificity ({config['metrics']['specificity']:.2%})")

# Check 4: Coefficients close to zero
if hasattr(model, 'coef_'):
    max_coef = np.max(np.abs(model.coef_[0]))
    if max_coef < 0.01:
        issues_found.append("❌ Coefficients near zero - features not contributing")

print("\n7️⃣ DIAGNOSIS SUMMARY")
print("-"*70)

if issues_found:
    print("\n🚨 PROBLEMS DETECTED:\n")
    for issue in issues_found:
        print(issue)
    
    print("\n💡 RECOMMENDATIONS:")
    print("""
    1. **Retrain the model** with proper class balancing:
       - Use SMOTE, class_weight='balanced', or stratified sampling
    
    2. **Check feature encoding**:
       - Ensure categorical variables are properly encoded
       - Verify feature scaling if needed
    
    3. **Verify training data**:
       - Check if you have both suppressed AND non-suppressed cases
       - Ensure adequate sample size for both classes
    
    4. **Try different algorithms**:
       - Random Forest (less sensitive to imbalance)
       - XGBoost with scale_pos_weight
       - SVM with class_weight='balanced'
    
    5. **Feature engineering**:
       - Add interaction terms
       - Create derived features (e.g., CD4 categories)
    """)
else:
    print("✅ No obvious issues detected")
    print("\nHowever, if the web app still predicts only one class, check:")
    print("1. Feature names match exactly between training and prediction")
    print("2. Feature values are in the same scale/encoding")
    print("3. Label encoders are being applied correctly")

print("\n8️⃣ FEATURE MAPPING CHECK")
print("-"*70)
print("Expected features by model:")
for i, feat in enumerate(config['features'][:10], 1):
    print(f"  {i}. {feat}")
if len(config['features']) > 10:
    print(f"  ... and {len(config['features']) - 10} more")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)