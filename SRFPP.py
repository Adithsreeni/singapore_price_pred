import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Set page configuration
st.set_page_config(page_title="Singapore Resale Flat Prices Prediction",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Set background
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://wallpapers.com/images/high/the-merlion-of-singapore-az393hiwvp29t2uz.webp");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)

setting_bg()

# Sidebar
st.sidebar.header(" :green[**Hi this is my dashboard**]")
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Price Prediction"],
                           icons=['house', 'coin'],
                           menu_icon="menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "#00ff00"},
                                   "nav-link-selected": {"background-color": "#ffa500"}})

if selected == 'Home':
    st.title("Welcome")
    col1, col2 = st.columns([1, 0.5], gap='small')
    with col1:
        st.write('### :red[Project Name]: Singapore Resale Flat Prices Prediction')
        st.write('### :red[Technologies Used]: Python, Data Preprocessing, EDA, Streamlit')
        st.write('### :red[Domain]: Real Estate')
    with col2:
        st.write("### :red[Overview]: The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore.")
else:
    col1, col2, col3 = st.columns(3)
    with col2:
        st.title(':blue[Price Prediction]')

    storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
                            '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
                            '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
                            '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
                            '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']

    with st.form('form'):
        col1, col2 = st.columns(2, gap='large')
        with col1:         
            month = st.selectbox('Month', options=[i for i in range(1, 13)])
            town = st.selectbox('Town', options=[i for i in range(0, 27)])
            flat_type= st.selectbox('Flat_type', options=[i for i in range(0, 7)])
            block= st.selectbox('Block', options=[i for i in range(1, 1000)])
            street_name= st.selectbox('Street_name', options=[i for i in range(0, 584)])
            
    
        with col2:
            storey_range = st.selectbox('Storey_range', options=storey_range_options)
            floor_area_sqm = st.selectbox('Floor_area_sqm', options=[i for i in range(28, 307)])
            flat_model= st.selectbox('Flat_model', options=[i for i in range(1, 13)])
            lease_commence_date = st.selectbox('Lease_commence_date', options=[i for i in range(1966, 2023)])
            year = st.selectbox('Reg_year', options=[i for i in range(1990, 2023)])   
            predict = st.form_submit_button('Predict selling price')

        if predict:
            try:
                with st.spinner('Getting Price'):
                    with open(r'E:\Python\SRFPP\dt.pkl', 'rb') as f:
                        model = pickle.load(f)

                    # Debug: Confirm model loaded
                    st.write("Model loaded successfully.")

                    # Define the encoder
                    ordinal_encoder = OrdinalEncoder(categories=[storey_range_options])
                    
                    # Encode categorical features using ordinal encoder
                    categorical_features = np.array([[storey_range]])
                    ordinal_encoder.fit(categorical_features)
                    categorical_encoded = ordinal_encoder.transform(categorical_features)
                    
                    # Debug: Show encoded features
                    st.write(f"Encoded storey_range: {categorical_encoded}")

                    # Prepare input data in the specified order
                    input_data = np.array([[month, town, flat_type, block, street_name, *categorical_encoded[0], flat_model, floor_area_sqm, lease_commence_date, year]])
                    
                    # Debug: Show prepared input data
                    st.write(f"Prepared input data: {input_data}")

                    # Predict the price
                    prediction = model.predict(input_data)[0]

                    # Debug: Show the prediction
                    st.write(f"Prediction: {prediction}")

                    # Decode the storey range (for demonstration purposes)
                    decoded_storey_range = ordinal_encoder.inverse_transform(categorical_encoded)
                    
                    # Debug: Show the decoded storey range
                    st.write(f"Decoded storey_range: {decoded_storey_range}")

                    st.markdown('### The Selling Price is')
                    st.markdown(f'# $:red[{prediction:,}]')
                    
                                
            except Exception as e:
                st.error(f"An error occurred: {e}")