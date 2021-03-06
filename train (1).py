from sklearn.ensemble import RandomForestClassifier 
import argparse 
import os 
import numpy as np 
from sklearn.metrics import mean_squared_error #
from sklearn.metrics import roc_auc_score
from sklearn import datasets #
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder #
import pandas as pd 
from azureml.core.run import Run 
from azureml.core import Workspace, Experiment 

def clean_data(data):
    # Clean data
    y_train = data['Survived']
    x_train = data.drop('Survived', axis=1)
    # Fill nulls in age with median
    x_train['Age'] = x_train.fillna(x_train['Age'].median())
    x_train["FamilySize"] = x_train["SibSp"] + x_train["Parch"]  
    x_train["WithFamily"] = 1
    x_train.loc[x_train["FamilySize"] > 0, "WithFamily"] = 0 
    x_train['Fare'] = x_train['Fare'].fillna(x_train['Fare'].median())
    dict_sex = {"male":0, "female":1}
    x_train['Sex']=df['Sex'].map(dict_sex)
    x_train.drop(['Cabin'], axis=1, inplace=True)
    
    return x_train, y_train

### Retrieve dataset ###

try :
    df = pd.read_csv("Downloads/titanic.csv")
except Exception:
    run = Run.get_context()
    ws = run.experiment.workspace

    found = False
    key = "titanic" 
    description_text = "Titanic data from Kaggle"

    if key in ws.datasets.keys(): 
        found = True
        df = ws.datasets[key] 
        df = df.to_pandas_dataframe()

x, y = clean_data(df)

### Train and evaluate ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100.0, help="number of trees in the forest")
    parser.add_argument('--random_state', type=int, default=1, help="controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node")

    args = parser.parse_args()

    run.log("N-estimators:", np.float(args.n_estimators))
    run.log("Random state:", np.int(args.random_state))

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()
