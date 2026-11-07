import streamlit as st
import requests
import json
import time
import os

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add title and description
st.title("House Price Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        A simple MLOps demonstration project for real-time house price prediction
    </p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Square Footage slider
    st.markdown(f"<p><strong>Square Footage:</strong> <span id='sqft-value'></span></p>", unsafe_allow_html=True)
    sqft = st.slider("", 500, 5000, 1500, 50, label_visibility="collapsed", key="sqft")
    st.markdown(f"<script>document.getElementById('sqft-value').innerText = '{sqft} sq ft';</script>", unsafe_allow_html=True)
    
    # Bedrooms and Bathrooms in two columns
    bed_col, bath_col = st.columns(2)
    with bed_col:
        st.markdown("<p><strong>Bedrooms</strong></p>", unsafe_allow_html=True)
        bedrooms = st.selectbox("", options=[1, 2, 3, 4, 5, 6], index=2, label_visibility="collapsed")
    
    with bath_col:
        st.markdown("<p><strong>Bathrooms</strong></p>", unsafe_allow_html=True)
        bathrooms = st.selectbox("", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], index=2, label_visibility="collapsed")
    
    # Location dropdown
    st.markdown("<p><strong>Location</strong></p>", unsafe_allow_html=True)
    location = st.selectbox("", options=["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"], index=1, label_visibility="collapsed")
    
    # Year Built slider
    st.markdown(f"<p><strong>Year Built:</strong> <span id='year-value'></span></p>", unsafe_allow_html=True)
    year_built = st.slider("", 1900, 2025, 2000, 1, label_visibility="collapsed", key="year")
    st.markdown(f"<script>document.getElementById('year-value').innerText = '{year_built}';</script>", unsafe_allow_html=True)
    
    # Price per Square Foot slider
    st.markdown(f"<p><strong>Expected Price per Sq Ft:</strong> <span id='price-per-sqft-value'></span></p>", unsafe_allow_html=True)
    
    # Valores por defecto basados en ubicación
    default_price_per_sqft_map = {
        "Rural": 180,
        "Suburb": 320,
        "Urban": 280,
        "Downtown": 350,
        "Waterfront": 450,
        "Mountain": 250
    }
    
    default_price_per_sqft = default_price_per_sqft_map.get(location, 300)
    price_per_sqft = st.slider("", 100, 800, default_price_per_sqft, 10, label_visibility="collapsed", key="price_per_sqft")
    st.markdown(f"<script>document.getElementById('price-per-sqft-value').innerText = '${price_per_sqft}/sq ft';</script>", unsafe_allow_html=True)
    
    # Condition dropdown
    st.markdown("<p><strong>Condition</strong></p>", unsafe_allow_html=True)
    condition = st.selectbox("", options=["Poor", "Fair", "Good", "Excellent"], index=2, label_visibility="collapsed")
    
    # Predict button
    predict_button = st.button("Predict Price", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    
    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("Calculating prediction..."):
            # Record start time for actual prediction time calculation
            start_time = time.time()
            
            api_data = {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location,
                "year_built": year_built,
                "condition": condition,
                "price_per_sqft": price_per_sqft
            }
            
            try:
                # Get API endpoint from environment variable or use default
                api_endpoint = os.getenv("API_URL", "http://localhost:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"
                
                # Make API call to FastAPI backend
                response = requests.post(predict_url, json=api_data)
                response.raise_for_status()
                prediction = response.json()
                
                # Calculate actual prediction time
                end_time = time.time()
                actual_prediction_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
                
                # Store prediction in session state with actual timing
                st.session_state.prediction = prediction
                st.session_state.actual_prediction_time = actual_prediction_time
                st.session_state.prediction_timestamp = end_time
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning("Please check your API connection and try again.")
                # Don't use mock data - let user know there's an actual error
                if "prediction" in st.session_state:
                    del st.session_state.prediction
    
    # Display prediction if available
    if "prediction" in st.session_state:
        pred = st.session_state.prediction
        
        # Format the predicted price
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f'<div class="prediction-value">{formatted_price}</div>', unsafe_allow_html=True)
        
        # ✅ Calculate confidence score dynamically from price range
        price_range = pred["confidence_interval"][1] - pred["confidence_interval"][0]
        confidence_percentage = max(60, min(95, int(100 - (price_range / pred["predicted_price"] * 100))))
        
        # ✅ Get model info from API response or detect from model file
        model_name = "Random Forest"  # This could come from API response in the future
        
        # Display confidence score and model used
        col_a= st.columns(1)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<p class="info-label">Confidence Score</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="info-value">{confidence_percentage}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display price range and actual prediction time
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
            lower = "${:,.0f}".format(pred["confidence_interval"][0])
            upper = "${:,.0f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            # ✅ Use actual prediction time
            actual_time = st.session_state.get('actual_prediction_time', 0)
            st.markdown(f'<p class="info-value">{actual_time} ms</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ✅ Dynamic top factors based on feature importance (if available)
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown("<p><strong>Top Factors Affecting Price:</strong></p>", unsafe_allow_html=True)
        
        # If API provides feature importance, use it; otherwise show general factors
        if pred.get("features_importance") and any(pred["features_importance"].values()):
            importance_items = sorted(pred["features_importance"].items(), 
                                    key=lambda x: x[1], reverse=True)[:4]
            factors_html = "<ul>"
            for feature, importance in importance_items:
                factors_html += f"<li>{feature.replace('_', ' ').title()} ({importance:.1%})</li>"
            factors_html += "</ul>"
            st.markdown(factors_html, unsafe_allow_html=True)
        else:
            # Fallback to logical factors based on input
            st.markdown("""
            <ul>
                <li>Square Footage (Primary driver)</li>
                <li>Location Type (Market factor)</li>
                <li>Price per Square Foot (Market rate)</li>
                <li>Property Condition</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ✅ Show prediction timestamp
        if "prediction_timestamp" in st.session_state:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", 
                                    time.localtime(st.session_state.prediction_timestamp))
            st.markdown(f'<p style="color: #6b7280; font-size: 12px; text-align: center;">Predicted at: {timestamp}</p>', 
                       unsafe_allow_html=True)
    else:
        # Display placeholder message
        st.markdown("""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Course</strong></p>
        <p>by WJM</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)