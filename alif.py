

import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("credit_train.csv")

print(df)


df["credit_sum"] = df["credit_sum"].replace(",", ".").astype(float)
df["score_shk"] = df["score_shk"].replace(",", ".").astype(float)


df.drop(["client_id", "living_region"], axis=1, inplace=True)


df = pd.get_dummies(df)


X = df.drop("open_account_flg", axis=1)
y = df["open_account_flg"]

from sklearn.impute import Imputer

imputer = Imputer(strategy='mean')
imputed_X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

