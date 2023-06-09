import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split

df = pd.read_csv("final dataset.csv")

x = df.drop(['HeartDisease'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Define the pre-trained model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(17,)))  # Modify the input shape accordingly
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Function to load and use the pre-trained model
def predict_heart_disease_percentage(data, model):
    predicted_percentage = model.predict(data)
    return predicted_percentage

def run_website():
    with st.sidebar:
        selected = option_menu('PuTongPuTong', ['Homepage', 'Prediction'],
                               default_index=0)

    if (selected == 'Homepage'):
        st.title('Heart Disease Prediction')

    if (selected == 'Prediction'):
        # Set the title and description of the app
        st.title('Heart Disease Prediction')
        st.markdown('Enter the necessary information to predict the percentage of getting heart disease.')

        # Create input fields for the user
        weight = st.number_input('Please enter your weight in kilograms')
        height = st.number_input('Please enter your height in centimeters:')

        # Calculate BMI if height is non-zero
        bmi = None
        if height != 0:
            height_m = height / 100  # Convert height to meters
            bmi = weight / (height_m ** 2)

        # Display the calculated BMI if available
        if bmi is not None:
            st.write(f'BMI: {bmi:.2f}')

        # User input for Smoking
        smoking_options = ['No', 'Yes']
        smoking = st.selectbox('Are you a current smoker?', smoking_options)

        # User input for Alcohol Drinking
        alcohol_drinking_options = ['No', 'Yes']
        alcohol_drinking = st.selectbox('Do you consume alcohol?', alcohol_drinking_options)

        # User input for Stroke
        stroke_options = ['No', 'Yes']
        stroke = st.selectbox('Have you experienced a stroke in the past?', stroke_options)

        # User input for Physical Health
        physical_health = st.slider('Rate your physical health on a scale of 1 to 30', 0, 30, step=1)

        # User input for Mental Health
        mental_health = st.slider('Rate your mental health on a scale of 1 to 30', 0, 30, step=1)

        # User input for Difficulty Walking
        walking_options = ['No', 'Yes']
        diff_walking = st.selectbox('Do you experience difficulty in walking?', walking_options)

        # User input for Sex
        sex_options = ['Male', 'Female']
        sex = st.selectbox('Select your gender', sex_options)

        # User input for Age Category
        # List of available age categories
        age_categories = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                          '65-69', '70-74', '75-79', '80 or older']
        age_category = st.selectbox('Select your age category', age_categories)

        # User input for Diabetic
        diabetic_options = ['No', 'Yes']
        diabetic = st.selectbox('Do you have diabetes?', diabetic_options)

        # User input for Physical Activity
        physical_activity_options = ['No', 'Yes']
        physical_activity = st.selectbox('Do you engage in regular physical activity?', physical_activity_options)

        # User input for General Health
        genhealth_options = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
        gen_health = st.selectbox('Rate your general health?', genhealth_options)

        # User input for Sleep Time
        sleep_time = st.slider('How many hours of sleep do you get per night?', 0, 24, step=1)

        # User input for Asthma
        asthma_options = ['No', 'Yes']
        asthma = st.selectbox('Do you have asthma?', asthma_options)

        # User input for Kidney Disease
        kidney_disease_options = ['No', 'Yes']
        kidney_disease = st.selectbox('Do you have kidney disease?', kidney_disease_options)

        # User input for Skin Cancer
        skin_cancer_options = ['No', 'Yes']
        skin_cancer = st.selectbox('Have you been diagnosed with skin cancer?', skin_cancer_options)

        # User input for Race
        race_options = ['American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White']
        race = st.selectbox('Please indicate your ethnic background', race_options)

        # Convert user inputs to appropriate values
        smoking = 1 if smoking == 'Yes' else 0
        alcohol_drinking = 1 if alcohol_drinking == 'Yes' else 0
        stroke = 1 if stroke == 'Yes' else 0
        diff_walking = 1 if diff_walking == 'Yes' else 0
        sex = 1 if sex == 'Male' else 0
        diabetic = 1 if diabetic == 'Yes' else 0
        physical_activity = 1 if physical_activity == 'Yes' else 0
        gen_health_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
        if gen_health in gen_health_mapping:
            gen_health = gen_health_mapping[gen_health]
        age_category_mapping = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6,
                                '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12}

        # Assigning the mapped value based on the condition
        if age_category in age_category_mapping:
            age_category = age_category_mapping[age_category]
        asthma = 1 if asthma == 'Yes' else 0
        kidney_disease = 1 if kidney_disease == 'Yes' else 0
        skin_cancer = 1 if skin_cancer == 'Yes' else 0

        # Create a button to trigger the prediction
        if st.button('Predict'):

            # Find the index of the selected race option
            race_index = race_options.index(race)

            # Create the user data array with all the features
            user_data = [bmi, smoking, alcohol_drinking, stroke, physical_health, mental_health, diff_walking, sex,
                         age_category, diabetic, physical_activity, gen_health, sleep_time, asthma, kidney_disease,
                         skin_cancer, race_index]

            # Convert the user data into a NumPy array or TensorFlow Tensor
            data = np.array(user_data).reshape(1, -1)  # Or use tf.constant() if working with TensorFlow

            print(data.shape)

            # Make predictions using the pre-trained model
            predicted_percentage = predict_heart_disease_percentage(data, model)

            # Display the predicted percentage of getting heart disease
            st.markdown('**Predicted Percentage of Getting Heart Disease:**')

            if predicted_percentage[0][0] >= 0:
                # Formatting the percentage
                formatted_percentage = "{:.1f}".format(predicted_percentage[0][0] * 100)

                # Displaying the formatted percentage
                st.write("Percentage:", formatted_percentage, "%")

run_website()

