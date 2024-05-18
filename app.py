import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Copper Modeling", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
def background():
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 2.5em;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.5em;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
        }
        .stButton>button {
            color: white;
            background-color: #007BFF;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)
    
background()

# Title and subtitle
st.markdown('<div class="title">Copper Modelling - Streamlit App</div>', unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Sidebar Menu")
st.sidebar.write("Use this sidebar to navigate through the app.")

option = st.sidebar.radio("Select an option", ['Status Prediction', 'Selling Price Prediction'])

# Initialize Std Scaler
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

item_type_map={'W':1, 'WI':2, 'S':3, 'Others':4, 'PL':5, 'IPL':6, 'SLAWR':7}
status_map= {'Won':1, 'Draft':2, 'To be approved':3, 'Lost':4, 'Not lost for AM':5,
            'Wonderful':6, 'Revised':7, 'Offered':8, 'Offerable':9}

if option=="Selling Price Prediction":


    
    st.markdown('<p style="color:blue;">PRICE PREDICTION</p>', unsafe_allow_html=True)

    with st.form("user_input_form"):

        # Splitting two colummns
        column_width=[2,0.5,2]
        col1,col2,col3=st.columns(column_width)

        with col1:    
           
            country=st.selectbox('**Country  Code**', ['28', '25', '3', '32', '38', '78', '27', '77', '113', '79', '26',
                                    '39', '4', '84', '8', '107', '89'])
            
            item_type=st.selectbox('**Item Type**', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'])
            
            application=st.selectbox('**Application**', ['1', '41', '28', '59', '15', '4', '38', '56', '42', '26', '27', '19', '2', '66', '29',
                                    '22', '25', '67', '79', '3', '99', '5', '39', '69', '7', '65', '58', '68'])
            
            product_ref=st.selectbox('**Product Referrence**', [1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                                        164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374,
                                        1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                                        1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                                        1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                                        1665584320, 1665584662, 1665584642])
            
            status=st.selectbox('**Status**', ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                                'Wonderful', 'Revised', 'Offered', 'Offerable'])
            

        with col3:

            customer=st.text_input('Enter Customer ID (7 & 8 Digits- starts with 30)')
            
            thickness=st.number_input('Enter Thickness')
            
            width=st.number_input('Enter Width')
            
            Quantity=st.number_input('Enter Quantity in Tonnage')



            # Every form must have a submit button.
            submitted = st.form_submit_button("Predict Selling Price")
            if submitted:
               

                item_type_map={'W':1, 'WI':2, 'S':3, 'Others':4, 'PL':5, 'IPL':6, 'SLAWR':7}
                status_map= {'Won':1, 'Draft':2, 'To be approved':3, 'Lost':4, 'Not lost for AM':5,
                            'Wonderful':6, 'Revised':7, 'Offered':8, 'Offerable':9}

                with open(r'C:\Users\Good Day\Desktop\Project 5\prediction.pkl', 'rb') as file:
                    model=pickle.load(file)

                # Map the categorical variables to their numerical values
                status = status_map[status]
                item_type = item_type_map[item_type]
                  
                # Prepare user input data
                user_input_data = np.array([[np.log(float(Quantity)), customer, country, item_type, application, 
                                            np.log(float(thickness)), width, product_ref, status]])
                # scaling data on user data
                st.write(user_input_data)
                # user_input_data=scaler.fit_transform(user_input_data)
                
                # Reshape input data to 2D array as expected by the model
                # user_input_data = user_input_data.reshape(1, -1)
                
                # Make prediction
                y_pred = model.predict(user_input_data)

                
                
                # Convert log prediction back to original scale
                selling_price=np.exp(y_pred)
                
                
                # Display the predicted selling price
                st.write(f"The predicted selling price is: {selling_price}")
                # st.write(user_input_data)
               
                
                # user_input_data= np.array([[np.log(float(Quantity)), customer, country, status, item_type, application,
                #                            np.log(float(thickness)), width, product_ref]])
                
                # # Assuming the model expects a 2D array with a single sample
                # user_input_data = user_input_data.reshape(1, -1)
                
                # y_pred=model.predict(user_input_data)

                # selling_price=np.exp(y_pred[0])

                # st.write(f"The predicted selling price is: {selling_price}")

# Status Prediction
if option=="Status Prediction":


    
    st.markdown('<p style="color:blue;">STATUS PREDICTION</p>', unsafe_allow_html=True)

    with st.form("price"):

        # Splitting two colummns
        column_width=[2,0.5,2]
        col1,col2,col3=st.columns(column_width)

        with col1:    
           
            # customer=st.text_in('**Customer ID**')
            country=st.selectbox('**Country  Code**', ['28', '25', '3', '32', '38', '78', '27', '77', '113', '79', '26',
                                    '39', '4', '84', '8', '107', '89'])
            
            item_type=st.selectbox('**Item Type**', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'])
            
            application=st.selectbox('**Application**', ['1', '41', '28', '59', '15', '4', '38', '56', '42', '26', '27', '19', '2', '66', '29',
                                    '22', '25', '67', '79', '3', '99', '5', '39', '69', '7', '65', '58', '68'])
            
            product_ref=st.selectbox('**Product Referrence**', [1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                                        164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374,
                                        1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                                        1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                                        1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                                        1665584320, 1665584662, 1665584642])
            

        with col3:

            customer=st.text_input('**Enter Customer ID** (7 & 8 Digits- starts with 30)')
            
            thickness=st.number_input('**Enter Thickness**')
            
            width=st.number_input('**Enter Width**')
            
            Quantity=st.number_input('**Enter Quantity in Tonnage**')

            selling=st.number_input('**Enter Selling Price**')


            # Every form must have a submit button.
            submitted = st.form_submit_button("Predict Status")
            if submitted:

            

                item_type_map={'W':1, 'WI':2, 'S':3, 'Others':4, 'PL':5, 'IPL':6, 'SLAWR':7}
                status_map= {'Won':1, 'Draft':2, 'To be approved':3, 'Lost':4, 'Not lost for AM':5,
                            'Wonderful':6, 'Revised':7, 'Offered':8, 'Offerable':9}

                with open(r'C:\Users\Good Day\Desktop\Project 5\classify.pkl', 'rb') as file:
                    model=pickle.load(file)

                # Map the categorical variables to their numerical values
                # status = status_map[status]
                item_type = item_type_map[item_type]
                  
                # Prepare user input data
                user_input_data = np.array([[np.log(float(Quantity)), customer, country, item_type, application, 
                                            np.log(float(thickness)), width, product_ref, np.log(float(selling))]])
                
                user_input_data=scaler.fit_transform(user_input_data)
                
                # Reshape input data to 2D array as expected by the model
                # user_input_data = user_input_data.reshape(1, -1)
                
                # Make prediction
                y_pred = model.predict(user_input_data)

                if y_pred==1:

                    st.write("The predicted Status is: Won")

                elif y_pred==4:

                    st.write("The predicted Status is: Lost")

                else:

                    st.write(f"The Predicted Status is: {y_pred}")
                
                
                    

            

        


