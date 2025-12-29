import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv("alzheimer.csv")
print(df.head())
df=df[["Age","MMSE","EDUC","Group"]]

df['Group']=df['Group'].map({
    'Non-Demented':0,
    'Demented':1
    })
print(df.head())

#Input
x=df[["Age","MMSE","EDUC"]]

#Output
y = df['Group'].to_numpy().ravel()


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

accuracy=model.score(x_train,y_train)
print("Accuracy of the model is:",accuracy)

pickle.dump(model, open("alzheimer_model.pkl", "wb"))
print("Model saved successfully!")