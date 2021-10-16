## Project: Disaster Response Pipelines 

### 1 - Overview

#### The main objective of this project is to analyze disaster data from [Figure Eight](https://appen.com/) and build a model, for an API, that classifies disaster messages.

### 2 - Project components

#### The web App file structure is:

#### - app
#### |- template
#### | | - master.html # the main page of web app
#### | | - go.html #the classification result page
#### | - run.py # Flask file that runs the app

#### - data
#### | - disaster_categories.csv # data to be processed
#### | - disaster_messages.csv # data to be processed
#### | - process_data.py # ETL script
#### | - DisasterResponse.db #database to save the cleaned data

#### - models
#### | - train_classifier.py # machine learning pipeline script
#### | - classifier.pkl # file with the saved model generated by train_classifier.py





### 3 - Instructions to access the app:

#### 1) download the 3 folders (app, data and models) into an empty new folder;

#### 2) with all 3 folders in same directory:

####

#### A
#### I - Execute process_data.py:

#### Go inside data directory(cd data) and type the following command in the terminal (inside data directory):


#### python process_data.py disaster_categories.csv disaster_messages.csv DisasterResponse.db

(This will generate DisasterResponse.db inside data directory)

####

#### B
#### II - Execute train_classifier.py to generate pickle file classifier.pkl:

#### - go to models folder and then, inside models directory (cd models), in the terminal, insert: 

#### i)npython train_classifier.py data/DisasterResponse.db classifier.pkl

#### or 

#### ii) put DisasterResponse.db inside models directory and simply type:

####  python train_classifier.py DisasterResponse.db classifier.pkl)

(This will generate classifier.pkl inside models directory)

###
#### C
#### 3) After generate the pickle file (inside models), classifier.pkl: go to app directory (cd app) and execute run.py:  

#### I- in the terminal insert the following command (inside app directory):

#### i) python run.py data/DisasterResponse.db models/classifier.pkl

#### or

#### ii) put classifier.pkl and DisasterResponse.db inside data directory and simply type:

#### python run.py DisasterResponse.db classifier.pkl

#### After that, a link is generated and it is possible to access the app.



### 4 - The trained model

#### There are 36 categories in which the messages will be classified.
#### The created model has an average precision of 0.76. 
#### However, the low number of samples in some categories in the trained data generates some problems in the classification accuracy. The high number of samples improves the prediction accuracy.  

### 5 - The Disaster Response Web App

![](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image1.png)

#### Overview of main page:

![Complete overview of main page](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image2.png)

#### Some classification examples:

![Message Classification: example](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image3.png)

![Another Classification example](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image4.png)

#### The classification could be improved with more data related with categories that have low number of samples; 
#### detected problem: different classification results for the same word - water:

![Some problems detected: example](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image5.png)
![Some problems detected: example ](https://github.com/a-teresa/disaster_response_pipelines/blob/main/screenshot_app/image6.png)



### 6 - Credits

#### -  The data used in the project is provided by [Figure Eight](https://appen.com/)

#### - The project is build under [Udacity](https://www.udacity.com) orientation (Data Science Nanodegree) both classroom and [Knowledge](https://knowledge.udacity.com/) space. 






