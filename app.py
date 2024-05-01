#importing all the important libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from propelauth import auth
from dotenv import load_dotenv
import os


#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Menu",["Home","Check Claim Integrity","Report a Fraudulent Claim", "Speak to a State Farm Agent"])

#Home Page 
user = auth.get_user()
if user is None:
    st.error('Unauthorized')
    st.stop()

with st.sidebar:
    st.link_button('Account', auth.get_account_url(), use_container_width=True)

#st.write("Logged in as " + user.email + " with user ID " + user.user_id)
s = user.email
st.success("Welcome " + s.split("@")[0] + "!")
#displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("State Farm Insurance Claims Hub")
    st.write("Welcome to the Neighborhood!")
    st.image("mustang_actual.png", "Verify the integrity of my vehicle claim")
    st.image("state-farm-newnew.png", "Report a fraudulent vehicle insurance claim")
    st.image("jake.png", "Speak to a State Farm insurance agent")


#loading the dataset
df = pd.read_csv("carclaims.csv")
df.drop(['Make','WeekOfMonth','DayOfWeek','DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed','MaritalStatus','PolicyNumber','Days:Policy-Accident','AddressChange-Claim','Year',],axis = 1, inplace = True)

df['AgeOfVehicle'].unique()
df['PastNumberOfClaims'].unique()
df['AgeOfPolicyHolder'].unique()
df['AgeOfVehicle'].unique()
df['VehiclePrice'].unique()
df['NumberOfCars'].unique()
df['PastNumberOfClaims'] = df['PastNumberOfClaims'].map({'none':1,'1':1,'2 to 4':4 , 'more than 4':5})
df['AgeOfVehicle'] = df['AgeOfVehicle'].map({'3 years':3,'6 years':6,'7 years':7,'more than 7':8,'5 years':5,'new': 0, '4 years':4, '2 years':2})
df['AgeOfPolicyHolder'] = df['AgeOfPolicyHolder'].map({'26 to 30':2,'31 to 35':3,'41 to 50':4,
                                                       '51 to 65':6,'21 to 25':1,'36 to 40':5,'16 to 17':0,'over 65':7,'18 to 20':0})

df['NumberOfCars'].unique()
df['FraudFound'] = df['FraudFound'].map({'yes':1,'No':0})
df.drop(['Month'] , axis = 1, inplace = True)
df.drop(['AccidentArea'],axis = 1, inplace = True)
df.drop(['VehicleCategory'],axis = 1, inplace= True)
df1 = pd.get_dummies(df, columns = ['Sex','PoliceReportFiled','Fault','PolicyType','WitnessPresent','AgentType','BasePolicy'],drop_first=True)

df1['Days:Policy-Claim'].unique()
df1['Days:Policy-Claim'] = df['Days:Policy-Claim'].map({'more than 30':35,'15 to 30':25,'8 to 15':12,'none':0})
df1.drop(['NumberOfSuppliments'],axis = 1, inplace=True)
df1.drop(['VehiclePrice'],axis = 1, inplace=True)
df1.drop(['NumberOfCars'],axis = 1, inplace=True)
df1['FraudFound']=df1['FraudFound'].fillna(value = 1)

from sklearn.model_selection import train_test_split

train = df1.drop('FraudFound',  axis = 1)
test = df1['FraudFound']

x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = 0.19, random_state = 20)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)
prediction = logreg.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, prediction))

#Verify Claim Integrity Page

if rad=="Check Claim Integrity":
    st.header("Know If Your Claim Report is Authentic")
    st.write("All The Values Should Be In Range Mentioned")
    
    age=st.number_input("Enter your age",min_value=0,max_value=150,step=1)
    repnumber=st.number_input("Enter representative num",step=1)
    deductible=st.number_input("Enter Your deductible",step=1)
    driverrating=st.number_input("Enter the driver rating",step=1)
    dayspolicyclaim=st.number_input("Enter your number of days since claim filed",step=1)
    pastnumclaims=st.number_input("Enter your past num claims",step=1)
    vehicleage=st.number_input("Enter your vehicle's age",step=1)
    policyholderage=st.number_input("Enter the policy holder's age",step=1)
    #fraudfound=st.number_input("Enter if fraud found",step=1)
    sex_male=st.number_input("Enter 1 if you are male (0 otherwise)",step=1)
    policereportfiled=st.number_input("Enter if policy report filed",step=1)
    faultthirdparty=st.number_input("Enter if fault of third party",step=1)
    sedancollision=st.number_input("Enter if sedan collision",step=1)
    sedanliability=st.number_input("Enter if sedan liability",step=1)
    sportall=st.number_input("Enter sports all",step=1)
    sportcollision=st.number_input("Enter sports collision",step=1)
    sportliability=st.number_input("Enter sports liability",step=1)
    utilityall=st.number_input("Enter if any utility",step=1)
    utilitycollision=st.number_input("Enter utility collision",step=1)
    utilityliability=st.number_input("Enter utility liability",step=1)
    witnesspresent=st.number_input("Enter if witness present",step=1)
    agentpresent=st.number_input("Enter if agent present",step=1)
    policycollision=st.number_input("Enter if policy collision",step=1)
    policyliability=st.number_input("Enter if policy liability",step=1)
    
    user_data = [age,repnumber,deductible,driverrating,dayspolicyclaim,pastnumclaims,vehicleage,policyholderage,sex_male,policereportfiled,faultthirdparty,sedancollision,sedanliability,sportall,sportcollision,sportliability,utilityall,utilitycollision,utilityliability,witnesspresent,agentpresent,policycollision,policyliability]
#for col in x_test.columns:
#  user_data.append(x_test.loc[403, col]) # filling user array, assuming they inputted values from x_test row 403 (random row)

    combined_data = []
    combined_data.append(user_data)
    #print(combined_data)

    np_data = np.array(combined_data)
    #print(np_data.shape)
    user_df = pd.DataFrame(data=np_data, columns=x_test.columns)
    #user_df

    y_value = y_test.loc[403]
    user_value = []
    user_value.append(y_value)
    #print(user_value)
    value = []
    value.append(user_value)
    #print(value)
    np_value = np.array(user_value)
    #print('shape', np_value.shape)
    value_df = pd.DataFrame(data=np_value)
    #value_df

    
    prediction = logreg.predict(np_data)
#accuracy_score(value_df, prediction)
    #prediction3=logreg.predict([[age,repnumber,deductible,driverrating,pastnumclaims,vehicleage,policyholderage,fraudfound,sex_male,policereportfiled,faultthirdparty,sedancollision,sedanliability,sportall,sportcollision,sportliability,utilityall,utilitycollision,utilityliability,witnesspresent,agentpresent,policycollision,policyliability]])[0]
    
    if st.button("Predict"):
        #print(prediction)
        if prediction[0]==0:
            st.success("Your claim is most likely authentic")
        else:
            st.warning("Your claim is likely to be fraudulent")

            
            errors = np.array([])
            columns = np.array(train.columns)

            for col in columns:
                mean = float(train.loc[:, col].mean())
                data = float(user_df[col])
                error = ((abs(data - mean)) / mean) * 100
                if (float(error) < 100):
                    errors = np.append(errors, float(error))
                else:
                    errors = np.append(errors, float(-1))
            
            max = np.max(errors)
            max_index = 0

            for i in range(len(errors)):
                if (errors[i] == max):
                    max_index = i
            
            col_val = columns[max_index]

            st.markdown("Factor likely responsible for fraudulent claim: " + col_val)
            st.write("Generating AI-driven recommendations to handle fraudulent claims...")

            import pathlib
            import textwrap

            import google.generativeai as genai

            from IPython.display import display
            from IPython.display import Markdown


            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
            
            load_dotenv()
            API_KEY = os.getenv("API_KEY")
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            prompt = "what are some advice for an insurance agent on steps to take after receiving a fraudulent insurance claim? and how can they prevent fraudulent insurance claims?"
            response = model.generate_content(prompt)
            st.markdown(response.text)
                                                    
