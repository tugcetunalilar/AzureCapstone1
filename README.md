# Surviving Titanic

This project utilises the Titanic-dataset in Kaggle to predict whether a passenger would survive or not. This is a classification task I used the RandomForestClassifier with hyperdrive and also the AutoML in order to find the best possible model. Both received similar results. As the AutoML received a slightly better accuracy, I deployed the model and also consumed the endpoint by sending a datapoint and receiving a prediction. I have taken several screenshots of different aspects of this project. They can be found in the screencasts folder. I will not bring all here to keep readability high.

## Project Set Up and Installation

## Setup
For this project, one needs a workspace in Azure for machine learning. In the workspace:
0. Download the code as a zip-file and extract all files from the zip.
1. Download the workspace config. In the image below, clicking on the workspace name, where the red arrow points, opens a panel on the left side, where one can download the config file highlighted in the image below with a red circle.
![where to find the config](/screenshots/get_config.png)

2. Go to notebooks and upload both .ipynb files, score.py, train.py as well as titanic.csv from these files and the config.json file downloaded in the previous step.
3. import the titanic-dataset into datasets
4. Create a compute named "titanic-compute" and a compute cluster named "ML-cluster-uda"
5. Run the cells in the notebooks.

## Dataset

### Overview
This is the titanic-dataset available in Kaggle. In the image below you can see the top rows of the dataset before it has been cleaned. It has information on the sex, age, family situation on board, class on board, cabin, embarkment etc.
![columns in the raw data](/screenshots/raw_data.png)

### Cleaning data
The data was cleaned. I filled the missing information in the age column with the medium value of that column. I transformed the values male and female to 0 and 1 respectively. I transformed information on the family size and created the feature "isAlone" to highlight those without family on board. I dropped the columns name, ticket, cabin and embarked. An example of the clean data is below:
![columns in the clean data](/screenshots/clean_data.png)

### Task
I used the clean data and all the features it included to solve a classifying problem whether each passenger survived or not. So the age, sex, number of siblings on board, parch, fare, family size and is alone are the features used. The label, survived, can be found in the image above as the column y.

### Access
I built a two way system to access the data. I use the uploaded csv and if that fails, I retrieve it from the workspace datasets.

## Automated ML
The settings I chose were designed to combine accuracy with also an idea of sparing resources. I chose a timeout of 20 minutes to make sure my experiments did not last too long. I chose the max concurrent iterations to 5 to limit the resources my experiments took. I chose the primary metric accuracy, because I could make it work for both automl and hyperdrive in the time I had. A better option would have been AUC. My task was a classification task. Early stopping as True is designed to save resources.

### Results
All the models received fairly similar results. The lowest accuracy was 0,6162 with run 25 utilising truncated SVCWrapper with XGBoostClassifier. Otherwise the models were fairly close to each other around 0,77-0,81 in accuracy. The VotingEnsemble often receives the best outcomes due to its ability to combine several different models and then they vote on the outcome therefore negating the possibly misleading results of any individual model. This was the case this time as well. The accuracy of the Voting Ensemble was 0,817. 
![best automlrun](/screenshots/best_automl_run_102.png)

The VotingEnsemble ensembled XGBoostclassifier, random forest, extreme random trees and LightGBM algorithms.
![best automl model](/screenshots/best_automl_model.png)

RunDetails widget:
![automl RunDetails](/screenshots/RunDetails_automl.png)

Best model with its parameters:
![best model](/screenshots/best_automl_model.png)

For more information on this model, please see the automl.ipynb

How to improve this: To improve this, I would work with the data more to bring more finesse to the handling of the data. I would also use AUC as the primary metric as that is often a better metric than Accuracy.


## Hyperparameter Tuning

I chose the RandomForestClassifier because it is very robust method to use in classifying problems. It is a meta estimator, an ensamble method, that combines several decision trees and then combines their classification and chooses the one that most trees agree upon. It is also easy to use whereas a voting ensemble or similar other ensemble methods need more configuration and coding.

I chose the random parameter sampling as I discovered in my studies that it receives similar results to a grid search. I chose as hyperparameters: 
* the n-estimators, which is the number of trees in the forest. For this the options were 5, 10, 15, 20, 25.
* max-depth, which is the depth of each tree. I chose the range 0f 3-10 for this hyperparameter.
* min-samples-split, which is the minimal samples required in each split. I chose as options 2, 5, 8, 12, 15, 18. 
* random-state, which controls the randomness of the bootstrapping of samples for the trees. For this hyperparameter, the options were a range from 1-10.

My main metric was Accuracy and the attempt was to maximize it. I also limited the amount of concurrent runs and the amount of runs as a way to affect the amount of resources needed.

My early-termination-policy was bandit policy. The slack-factor is 0.1, because I wished to limit the runs where the metric would be further from the best than that 0.1. I chose the evaluation interval to be 1 to assess the runs fairly frequently, but delayed the evaluation to 5 to gather some data before applying the termination policy.

### Results
The hyperdrive got very similar results to the automl-run. The accuracy was around the same. The best model was run 40, with max-depth of 7, min-samples split of 18, n-estimators 15 and random state 8. 

A visual on the hyperparameters of top 10 runs:
![top 10 hyperparameters and accuracies](/screenshots/hyperdrive_parameters_visual.png)

RunDetails widget:
![hyperdrive RunDetails](/screenshots/rundetails_hyperdrive.png)

Best model with its parameters:
![best hyperdrive model](/screenshots/best_hyperdrive_model.png)

For more information on this model, please see the hyperparameter_tuning.ipynb

How to improve this: To improve this, I would work with the data more to bring more finesse to the handling of the data. I would also use AUC as the primary metric as that is often a better metric than Accuracy.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
In the deployment, I utilised one ready-made environment in Azure for convenience and ease of use for other users. The environment I chose was "AzureML-AutoML". This does bring about the possibility of the model at some point not working due to changes in the environment. So one point of improvement of this project would be to create a specific environment for this project.
I deployed this as a ACI webservice and enabled logging to be able to see any issues. I also turned on application insights as seen below:
![application insights](/screenshots/application_insights.png)

To use this endpoint, once the project runs and the best model is deployed, one needs data in the shape seen below.
![data for endpoint](/screenshots/data_for_endpoint.png)

Then one needs the rest-api address beginning with http:// as seen below cell 76 of automl.ipynb. This api will be different when this project is run as the api URI is created when the service is deployed and the current one has been deleted. So the address needs to be copied to cell 80 where there is a variable scoring_uri. Then it is possible to run that cell and use the endpoint to get a prediction. It is also possible to copy the contents of cell 80 to a script and run a prediction from bash/terminal. In that case the endpoint URI will also need to be copied to the script. In that case, please also copy all necessary imports from the first cell in the notebook as some libraries will need to be imported for the cell 80 script to work.

For a demonstration of the working of this project and endpoint, please see the video in the link below.

Here's a screenshot of the endpoint being healthy and working.
![healthy endpoint](/screenshots/deployed_healthy_endpoint.png)

## Screen Recording
A video-demonstration of the project can be found here on youtube: https://youtu.be/M1EXbkMelDc

## Points of further development
All work can be improved upon and all improvements are further learning. However, as all projects need to be completed, some improvements are left outside the scope of a project. Here I have stated some areas of improvement that I would still like to do, but due to time constraints, must leave outside the scope of this project at this time of assessment.
1. using the metric AUC
2. More finesse in the handling of the data
3. Enable Swagger for documentation
4. Create an environment specifically for this project.

## Standout Suggestions
I enabled logging. This enables me to see, what is happening in the service. As seen below, one can see the requests, their type and time. Logs also show any errors.
![logs](/screenshots/logs.png)
