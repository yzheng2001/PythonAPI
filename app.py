import json
from typing import List
from fastapi import FastAPI, Depends
import pandas as pd
# more on using pandas to stats/CRUD the dataframe "data" (in memory table) after read_csv, https://www.youtube.com/watch?v=tRKeLrwfUgU
# more advanced pandas inbuilt methods to avoid loop etc on all kinds of t-sql type of query/stats/plot/conditional-formatting, https://www.youtube.com/watch?v=_gaAoJBMJ_Q
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

app = FastAPI()
print("Hello")
from utils.KidsDetailService import KidsDetailsService

@app.get("/api/hello")
async def get_kids_detail():
    service = KidsDetailsService()
    kids_details = await service.get_KidsDetails()
    return kids_details #"hello" 

@app.get("/api/jsonfile")
def get_jsonfile():
    # data = pd.read_json('data.json', orient="columns")
    # print(data)
    # # pandas function returns dataframe (table) or dataserie (column/row)
    # # Here, the data is stored in a columnar manner, with each key-value pair corresponding to column headers and values. {name:["myname","yourname]", age:[46,48]}
    # # When data is stored as a list of dictionaries, orient should be set to "records". {[{name:"myname", age:46},{name:"yourname", age:48}]}
    # # "index" is passed to orient when key-value pairs are made up of indicies and a dictionary containing the remainder of the record.
    # # {"myname":{age: 46, gender :"female"}, "yourname":{age : 48, gender:"male"}}
    # #https://campus.datacamp.com/courses/introduction-to-data-pipelines/advanced-etl-techniques?ex=1

    # raw_data=json.load("nested_data.json")
    # Opening JSON file 
    f = open('data.json',)    
    # returns JSON object as a dictionary 
    data = json.load(f) 
    print(data)
    #https://www.geeksforgeeks.org/json-load-in-python/?ref=ml_lbp
    return data

@app.get("/api/ai",  response_model=List[int])
def test_ai():

    # Step 2: Load the data (assuming it's in a CSV file)
    data = pd.read_csv('data.csv')

    # Step 3: Preprocess the data (handle missing values, encode categorical variables, etc.)
    # Example: Fill missing values with the mean
    data = data.fillna(data.mean())

    # Step 4: Split the data into features (X) and target variable (y)
    X = data.drop('has_disease', axis=1)
    y = data['has_disease']

    # below start to use the sklearn functionality to train (linear) model
    # Step 5: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Preprocessing (scaling the features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 7: Choose an algorithm (e.g., Logistic Regression)
    model = LogisticRegression()

    # Step 8: Train the model
    model.fit(X_train_scaled, y_train)

    # Step 9: Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Step 10: Make predictions on new data
    new_data =  [[51, 135, 140]] # Example new data point
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    print(f"Prediction for new data: {prediction}")
    return prediction



@app.get("/api/ai-2y", response_model=List[float])
async def test_ai_2y():
    # Load the data
    data = pd.read_csv('data2.csv')

    # Split the data into features (X) and target variables (y)
    X = data.drop(['has_disease', 'excerciseweekly'], axis=1)
    y = data[['has_disease', 'excerciseweekly']]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the features (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose an algorithm (e.g., Linear Regression for regression tasks)
    model = LinearRegression()

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse1 = mean_squared_error(y_test['has_disease'], y_pred[:, 0])
    mse2 = mean_squared_error(y_test['excerciseweekly'], y_pred[:, 1])
    print(f"MSE for has_disease: {mse1}")
    print(f"MSE for excerciseweekly: {mse2}")

    # Make predictions on new data
    new_data = [[51, 135, 140]]
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    print(f"Predictions for new data: {prediction}")
    result = prediction[0]
    return result
    #result =  json.dumps(prediction[0])
    #return result

@app.get("api/ai-2y-improved", response_model=List[int])
async def test_ai_2y_improved():
    # Load the data
    data = pd.read_csv('data2.csv')

    # Split the data into features (X) and target variables (y)
    X = data.drop(['has_disease', 'excerciseweekly'], axis=1)
    y1 = data['has_disease']
    y2 = data['excerciseweekly']

    # Split the data into training and test sets
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

    # Preprocess the features (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose algorithms (Linear Regression for continuous target, Logistic Regression for binary target)
    model1 = LinearRegression() 
    model2 = LogisticRegression()

    # Train the models
    model1.fit(X_train_scaled, y2_train)
    model2.fit(X_train_scaled, y1_train)

    # Evaluate the models
    y2_pred = model1.predict(X_test_scaled)
    y2_pred = y2_pred.round().astype(int)  # Round and convert to integer
    mse = mean_squared_error(y2_test, y2_pred)
    print(f"MSE for excerciseweekly: {mse}")

    y1_pred = model2.predict(X_test_scaled)
    y1_pred = y1_pred.round().astype(int)  # Round and convert to integer
    acc = accuracy_score(y1_test, y1_pred)
    print(f"Accuracy for has_disease: {acc}")

    # Make predictions on new data
    new_data =  [[51, 135, 140]]
    new_data_scaled = scaler.transform(new_data)
    prediction1 = model1.predict(new_data_scaled)
    prediction2 = model2.predict(new_data_scaled)
    print(f"Prediction for excerciseweekly: {int(round(prediction1[0]))}")
    print(f"Prediction for has_disease: {int(round(prediction2[0]))}")
    return "hello" #json.dump(int(round(prediction1[0])),int(round(prediction2[0])) )


@app.get("api/ai-2y-classification", response_model=List[int])
async def test_ai_2y_improved():
    # Load the data
    data = pd.read_csv('data_other.csv')

    # Split the data into features (X) and target variables (y)
    X = data.drop(['has_disease', 'excerciseweekly'], axis=1)
    y1 = data['has_disease']
    y2 = data['excerciseweekly']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test,= train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the features (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #MultinomialLogisticRegression: Use case: When the target variable is a categorical variable with more than two classes, i.e.multi-class classification  
    #e.g., classifying emails into multiple categories like spam, personal, work, etc.; classifying images into multiple categories like dogs, cats, birds, etc.
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #OneVsRestClassifier: Use case: When the target variable is multi-label (i.e., each sample can belong to multiple classes simultaneously).
    #Example scenario: Classifying a movie into multiple genres like action, comedy, drama, etc.
    base_estimator = LogisticRegression()
    model = OneVsRestClassifier(base_estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "hello"