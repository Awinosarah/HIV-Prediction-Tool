import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ============================================================
# FIX: Set correct base directory for model files
# ============================================================

model_path = 'final_best_model.pkl'
config_path = 'model_config.pkl'

# Page configuration
st.set_page_config(
    page_title="HIV Viral Load Prediction Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and configuration with FIXED PATHS
@st.cache_resource
def load_model_and_config():
    """Load the trained model and configuration from correct directory"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
        st.success(f"✅ Model loaded successfully from: {BASE_DIR}")
        return model, config
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ Model files not found!
        
        Looking for files at:
        - {MODEL_PATH}
        - {CONFIG_PATH}
        
        Please ensure the files exist in: {BASE_DIR}
        
        Error: {str(e)}
        """)
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, config = load_model_and_config()

# App Header
st.markdown('<h1 class="main-header">HIV/AIDS Viral Load Non-Suppression Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555;">Kawolo Hospital - Machine Learning-Based Risk Assessment</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Kawolo+Hospital", use_container_width=True)
    st.markdown("### About This Tool")
    st.info("""
    This tool uses machine learning to predict the risk of HIV viral load non-suppression 
    based on patient characteristics and clinical data.
    
    **Model Information:**
    - Algorithm: """ + (config['model_name'] if config else "N/A") + """
    - Technique: """ + (config['technique'] if config else "N/A") + """
    - Accuracy: """ + (f"{config['metrics']['accuracy']:.2%}" if config else "N/A") + """
    """)
    
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Enter patient information
    2. Click 'Predict Risk'
    3. Review results and recommendations
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** Research Team")
    st.markdown("**Institution:** Kawolo Hospital, Uganda")
    
    # Debug info (can be removed in production)
    with st.expander("🔧 Debug Info"):
        st.code(f"Base Dir: {BASE_DIR}\nModel exists: {os.path.exists(MODEL_PATH)}\nConfig exists: {os.path.exists(CONFIG_PATH)}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📖 Feature Guide", "ℹ️ About"])

with tab1:
    st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model not loaded. Cannot make predictions.")
        st.info(f"Please ensure model files exist in: {BASE_DIR}")
    else:
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Demographic Information")
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, help="Patient's age in years")
                gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
                age_category = st.selectbox("Age Category", ["Child (0-14)", "Youth (15-24)", "Adult (25-49)", "Older Adult (50+)"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed", "Unknown"])
                
            with col2:
                st.markdown("#### Clinical Information")
                cd4 = st.number_input("CD4 Count (cells/μL)", min_value=0, max_value=2000, value=350, help="Most recent CD4 count")
                who_stage = st.selectbox("WHO Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], help="Current WHO clinical stage")
                pregnancy_status = st.selectbox("Pregnancy Status", ["Not Pregnant", "Pregnant", "Not Applicable", "Unknown"])
                muac_status = st.selectbox("MUAC Status", ["Normal", "Moderate", "Severe", "Unknown"], help="Mid-Upper Arm Circumference")
                
            with col3:
                st.markdown("#### Treatment & Follow-up")
                total_visits = st.number_input("Total Clinic Visits", min_value=0, max_value=500, value=12, help="Total number of clinic visits")
                days_since_last = st.number_input("Days Since Last Appointment", min_value=0, max_value=365, value=30)
                next_appointment = st.number_input("Next Appointment (days)", min_value=0, max_value=365, value=30)
                lost_to_follow = st.selectbox("Lost to Follow-up Status", ["No", "Yes"], help="Has patient been lost to follow-up?")
                
            col4, col5 = st.columns(2)
            with col4:
                st.markdown("#### Treatment Details")
                art_start_date = st.date_input("ART Start Date", value=datetime(2020, 1, 1), help="Date antiretroviral therapy started")
                start_regimen = st.selectbox("Starting ART Regimen", ["TDF/3TC/EFV", "AZT/3TC/NVP", "ABC/3TC/EFV", "Other"])
                start_weight = st.number_input("Weight at ART Start (kg)", min_value=0.0, max_value=200.0, value=65.0)
                
            with col5:
                st.markdown("#### Additional Information")
                start_who_stage = st.selectbox("WHO Stage at ART Start", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
                patient_status = st.selectbox("Patient Status", ["Active", "Transferred Out", "Dead", "Lost to Follow-up"])
                combination_on = st.selectbox("Combination Therapy", ["Yes", "No"])
                care_entry = st.selectbox("Care Entry Point", ["VCT", "PMTCT", "TB Clinic", "Outpatient", "Other"])
                ab_status = st.selectbox("AB Status", ["Positive", "Negative", "Unknown"])
            
            st.markdown("---")
            submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True, type="primary")
            
        if submitted:
            # Prepare input data
            input_data = {
                'age': age,
                'gender': 1 if gender == "Male" else 0,
                'age category': ['Child (0-14)', 'Youth (15-24)', 'Adult (25-49)', 'Older Adult (50+)'].index(age_category),
                'total visits': total_visits,
                'days since last appointment': days_since_last,
                'next appointment days': next_appointment,
                'lost to follow status': 1 if lost_to_follow == "Yes" else 0,
                'marital status': ['Single', 'Married', 'Divorced', 'Widowed', 'Unknown'].index(marital_status),
                'who stage': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'].index(who_stage),
                'cd4': cd4,
                'pregnancy status': ['Not Pregnant', 'Pregnant', 'Not Applicable', 'Unknown'].index(pregnancy_status),
                'status': ['Active', 'Transferred Out', 'Dead', 'Lost to Follow-up'].index(patient_status),
                'combination on': 1 if combination_on == "Yes" else 0,
                'muac status': ['Normal', 'Moderate', 'Severe', 'Unknown'].index(muac_status),
                'ab': ['Positive', 'Negative', 'Unknown'].index(ab_status),
            }
            
            # Create DataFrame with all features from config
            input_df = pd.DataFrame([input_data])
            
            # Add missing features with default values (0) if needed
            if config:
                for feature in config['features']:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                # Reorder columns to match training data
                input_df = input_df[config['features']]
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="prediction-box high-risk">', unsafe_allow_html=True)
                        st.markdown("### ⚠️ HIGH RISK")
                        st.markdown("**Non-Suppression Predicted**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-box low-risk">', unsafe_allow_html=True)
                        st.markdown("### ✅ LOW RISK")
                        st.markdown("**Suppression Predicted**")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence Score", f"{max(prediction_proba):.1%}")
                    st.metric("Non-Suppression Risk", f"{prediction_proba[1]:.1%}")
                    st.metric("Suppression Probability", f"{prediction_proba[0]:.1%}")
                
                with col3:
                    # Risk gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction_proba[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Non-Suppression Risk %"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clinical recommendations
                st.markdown("---")
                st.markdown('<h3 class="sub-header">Clinical Recommendations</h3>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 🏥 Immediate Actions Required")
                    st.markdown("""
                    - **Schedule urgent follow-up** within 1-2 weeks
                    - **Conduct adherence counseling** session
                    - **Review current ART regimen** for potential changes
                    - **Assess barriers** to treatment adherence
                    - **Consider viral load testing** if not recently done
                    - **Evaluate for drug resistance** if multiple failures
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Risk factors
                    st.markdown("### 📋 Contributing Risk Factors")
                    risk_factors = []
                    if cd4 < 350:
                        risk_factors.append(f"Low CD4 count ({cd4} cells/μL)")
                    if lost_to_follow == "Yes":
                        risk_factors.append("Lost to follow-up status")
                    if who_stage in ["Stage 3", "Stage 4"]:
                        risk_factors.append(f"Advanced WHO stage ({who_stage})")
                    if days_since_last > 60:
                        risk_factors.append(f"Long time since last visit ({days_since_last} days)")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.info("No specific high-risk factors identified from available data.")
                else:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ Routine Care Recommendations")
                    st.markdown("""
                    - **Continue current ART regimen** as prescribed
                    - **Maintain regular follow-up** appointments
                    - **Schedule next viral load** as per protocol
                    - **Reinforce adherence** counseling at routine visits
                    - **Monitor for new symptoms** or side effects
                    - **Continue health education** and support
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Patient shows good predicted viral suppression. Continue current management plan.")
                
                # Key patient factors
                st.markdown("---")
                st.markdown('<h3 class="sub-header">Key Patient Factors</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Clinical Indicators:**")
                    st.write(f"- CD4 Count: {cd4} cells/μL")
                    st.write(f"- WHO Stage: {who_stage}")
                    st.write(f"- Total Visits: {total_visits}")
                    
                with col2:
                    st.markdown("**Follow-up Status:**")
                    st.write(f"- Days Since Last Visit: {days_since_last}")
                    st.write(f"- Lost to Follow-up: {lost_to_follow}")
                    st.write(f"- Patient Status: {patient_status}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all required fields are filled correctly.")
                with st.expander("See error details"):
                    st.code(str(e))

with tab2:
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    if config:
        st.markdown("### 📊 Overall Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{config['metrics']['accuracy']:.2%}", 
                     help="Overall percentage of correct predictions")
        with col2:
            st.metric("Precision", f"{config['metrics']['precision']:.2%}",
                     help="Of predicted non-suppressed cases, how many were actually non-suppressed")
        with col3:
            st.metric("Sensitivity", f"{config['metrics']['sensitivity']:.2%}",
                     help="Of actual non-suppressed cases, how many were correctly identified")
        with col4:
            st.metric("Specificity", f"{config['metrics']['specificity']:.2%}",
                     help="Of actual suppressed cases, how many were correctly identified")
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("F1-Score", f"{config['metrics']['f1_score']:.2%}",
                     help="Harmonic mean of precision and sensitivity")
        with col6:
            st.metric("AUC-ROC", f"{config['metrics']['auc_roc']:.2%}",
                     help="Overall discriminative ability of the model")
        
        # Visualize metrics
        st.markdown("---")
        st.markdown("### 📈 Performance Visualization")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC'],
            'Score': [
                config['metrics']['accuracy'],
                config['metrics']['precision'],
                config['metrics']['sensitivity'],
                config['metrics']['specificity'],
                config['metrics']['f1_score'],
                config['metrics']['auc_roc']
            ]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                     title='Model Performance Metrics',
                     color='Score',
                     color_continuous_scale='Viridis',
                     range_y=[0, 1])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        st.markdown("---")
        st.markdown("### 🤖 Model Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Algorithm:** {config['model_name']}
            
            **Balancing Technique:** {config['technique']}
            
            **Training Features:** {len(config['features'])} variables
            """)
        
        with col2:
            st.info(f"""
            **Data Preservation:** {config['data_preservation']['preservation_rate']}
            
            **Training Records:** {config['data_preservation']['preserved_rows']:,}
            
            **Original Records:** {config['data_preservation']['initial_rows']:,}
            """)
        
        # Feature list
        with st.expander("📋 View All Training Features"):
            st.write(", ".join(config['features']))
    else:
        st.warning("Model configuration not loaded.")

with tab3:
    st.markdown('<h2 class="sub-header">Feature Guide & Definitions</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This guide explains each feature used in the prediction model and how to collect accurate data.
    """)
    
    # Demographic features
    with st.expander("👥 Demographic Information", expanded=True):
        st.markdown("""
        **Age:** Patient's current age in years
        - Critical for age-stratified risk assessment
        - Age extremes (very young, elderly) may have different risk profiles
        
        **Gender:** Biological sex
        - Male/Female differences in adherence patterns
        - Impacts pregnancy status relevance
        
        **Age Category:** Grouped age ranges
        - Child (0-14 years)
        - Youth (15-24 years)
        - Adult (25-49 years)
        - Older Adult (50+ years)
        
        **Marital Status:** Current relationship status
        - May indicate social support availability
        - Influences treatment adherence
        """)
    
    # Clinical features
    with st.expander("🏥 Clinical Information"):
        st.markdown("""
        **CD4 Count:** CD4+ T-cell count (cells/μL)
        - Key immunological marker
        - Normal: >500 cells/μL
        - Low: <200 cells/μL (AIDS-defining)
        - Critical predictor of treatment outcomes
        
        **WHO Stage:** Clinical stage (1-4)
        - Stage 1: Asymptomatic
        - Stage 2: Mild symptoms
        - Stage 3: Advanced disease
        - Stage 4: Severe/AIDS-defining conditions
        
        **Pregnancy Status:** Current pregnancy state
        - Not Applicable (for males)
        - Pregnant/Not Pregnant (for females)
        - Affects treatment regimen selection
        
        **MUAC Status:** Mid-Upper Arm Circumference
        - Normal: >23.5 cm (adults)
        - Moderate malnutrition: 19-23.5 cm
        - Severe malnutrition: <19 cm
        - Nutritional status indicator
        """)
    
    # Follow-up features
    with st.expander("📅 Follow-up & Adherence"):
        st.markdown("""
        **Total Visits:** Cumulative clinic visits
        - Higher visits may indicate better engagement
        - Or higher visits due to complications
        - Context-dependent interpretation
        
        **Days Since Last Appointment:** Time elapsed
        - Shorter intervals: better engagement
        - >60 days: potential adherence issues
        - >90 days: high risk for non-suppression
        
        **Next Appointment Days:** Scheduled follow-up
        - Standard: 30-90 days for stable patients
        - Shorter for unstable/new patients
        
        **Lost to Follow-up Status:**
        - Yes: Missed appointments consistently
        - Strong predictor of non-suppression
        - Requires active tracing
        """)

with tab4:
    st.markdown('<h2 class="sub-header">About This Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    
    This web-based prediction tool was developed as part of a research study at Kawolo Hospital, Uganda, 
    to assist healthcare providers in identifying patients at risk of HIV viral load non-suppression.
    
    ### 🔬 Methodology
    
    The tool uses machine learning algorithms trained on historical patient data to predict the likelihood 
    of viral non-suppression based on demographic, clinical, and treatment-related factors.
    
    ### ⚕️ Clinical Application
    
    **This tool is designed to:**
    - Support (not replace) clinical judgment
    - Identify high-risk patients for targeted intervention
    - Optimize resource allocation
    - Improve patient outcomes through early intervention
    
    **This tool should NOT be used to:**
    - Make sole treatment decisions
    - Replace viral load testing
    - Override clinical expertise
    - Discriminate against patients
    
    ### ⚠️ Disclaimer
    
    This tool is provided for educational and research purposes. While the model has been validated 
    on historical data, predictions should always be interpreted in the context of comprehensive 
    clinical assessment. Healthcare providers remain responsible for all clinical decisions.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p><strong>Important:</strong> This tool provides decision support only. 
    Always consult with healthcare professionals for medical decisions.</p>
    <p style='font-size: 0.8rem;'>Developed for research and educational purposes | Kawolo Hospital, Uganda</p>
</div>
""", unsafe_allow_html=True)
