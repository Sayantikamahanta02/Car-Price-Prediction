import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#dataset load
df=pd.read_csv("car data.csv")
print(df)
print(df.head(4))

#dataset cleaning
print(df.info())
print(df.isnull().sum())
print(df.columns)
df['Car_Age'] = 2026 - df['Year']
print(df.columns)
print(df.isnull().sum())
print(df.shape)

#encode the catagorical data
car_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()
df['Car_Name'] = car_encoder.fit_transform(df['Car_Name'])
df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = seller_encoder.fit_transform(df['Seller_Type'])
df['Transmission'] = trans_encoder.fit_transform(df['Transmission'])
print(df.head(6))
print(df.tail(6))

#split the data
x=df.drop([ 'Selling_Price','Year'],axis=1)
y=df['Selling_Price']

#model train
model=RandomForestRegressor(n_estimators=14,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

#test accuracy
acc=r2_score(y_test,y_predict)
print("Accuracy score",acc)

#print available car name
print("Available Car Names:")
print(car_encoder.classes_)

# take user input
Car_Name = input("Enter the Car Name: ")
Car_Name = car_encoder.fit_transform([Car_Name])
Present_Price=float(input("Enter the Present_Price:"))
Kms_Driven=float(input("Enter the Kms_Driven:"))
Fuel_Type=(input("Enter the Fuel_Type:"))
Fuel_Type = fuel_encoder.fit_transform(['Fuel_Type'])
Seller_Type=(input("Enter the Seller_Type:"))
Seller_Type = seller_encoder.fit_transform(['Seller_Type'])
Transmission=(input("Enter the Transmission:"))
Transmission = trans_encoder.fit_transform(['Transmission'])
Owner=int(input("Enter the Owner:"))
Car_Age=int(input("Enter the Car_Age:"))

#make user input dataframe
data=pd.DataFrame([{
"Car_Name":Car_Name,
"Present_Price":Present_Price,
"Kms_Driven":Kms_Driven,
"Fuel_Type":Fuel_Type,
"Seller_Type":Seller_Type,
"Transmission":Transmission,
"Owner":Owner,
"Car_Age":Car_Age
}])

#predict the user input data 
final_result=model.predict(data)

#show the predicted output from the user input
print("Car selling price:",np.round(final_result,2))
