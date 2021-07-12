# Machine Learning Tutorial

*This repository tutorial for Machine learning preprocesses, Exploraty Data Analysis, Machine learning algorithms.*

# Contents
* What is the Machine Learning ?
* Learning methods
  * Supervised Learning
  * Unsupervised Learning
* Machine learning algorithms graphs and explanation
  * Regression models
  * Classification models
    * Supervised Learning
    * Unsupervised Learning
    * Introduction the Deep Learning with Logistic regression(a separate deep learning repository will be prepared)
* Encoding Types 
  * Label encoder
  * OneHotEncoding
* NLP(Natural Language Process)
* PCA(Principle Component Analysis)
* What is K-Fold-CrossValidation ? 
  * GridSearchCV vs RandomizedCV ?
* What are the Overfitting and Underfitting status ? 
* Recommendation Systems
* Exploraty Data Analysis and training data that we use

# What is the Machine Learning ?

**Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.**

*There are three types of learning methods in this repository.*
These are Supervised, Unsupervised, Ensemble methods.

## Supervised Learning
**Supervised Learning algorithms goal, if data has a label and we want to prediction these label we use supervised learning methods.**

Supervised learning methods also two type.
These are regression and classifacation.

*Regression algorithms goal is training data for continous labels.*
Regression types where given in below.
 * Linear Regression
 * Decision Tree Regression
 * Random Forest Regression

*Classification models goal train data for discrete labels and classification algorithms also two type Supervised and Unsupervised Learning algorithms.*
Classification types where given in below.
 * K neirest neighbor Classification
 * Linear SVM
 * Decision Tree Classification
 * Random Forest Classification
 * Naive Bayes Classification

## Unsupervised Learning
Unsupervised if data has a no label and we want to cluster to data we can use unsupervised learning method
  * KMeans
  * Hieararchical Clustering

# Machine learning algorithms graphs and explanation
## Regression models
### Linear Regression
![1548702778023](https://user-images.githubusercontent.com/51100947/123879620-e4d5f800-d949-11eb-87c6-98daa18d2e3f.png)

*Source for graph : https://www.jmp.com/en_us/statistics-knowledge-portal/what-is-multiple-regression/fitting-multiple-regression-model.html

The model usefull for continous labels 

*The goal is try to draw the most optimized line.*
*So why did the algorithm summing error and square? because some errors are positive value and samo errors are negative value, if we sum to the errors we might have errors zero and it's not realist value.*
*So the algorithms try to minimize MSE(mean squared eror).*

### Decision Tree 
![df](https://user-images.githubusercontent.com/51100947/123880125-b278ca80-d94a-11eb-89c9-56aaeda79190.png)

*Algorithm that the best try to splitting the coordinate plane into many parts(leaf we can say) and makes predictions as a result of comparisons.*

### Random forest 
![Fig-A10-Random-Forest-Regressor-The-regressor-used-here-is-formed-of-100-trees-and-the](https://user-images.githubusercontent.com/51100947/123880272-f79cfc80-d94a-11eb-83f7-2881a9780a9a.png)

* Source for graph : https://www.researchgate.net/figure/Fig-A10-Random-Forest-Regressor-The-regressor-used-here-is-formed-of-100-trees-and-the_fig3_313489088

*The algorithm usefull for recommendation algorithms(for example Netflix and YouTube recommendations)*
*Random forest actually, has a lot of decision trees's results that we avarage results.*
*Ensemble learning family that using multiple ml algorithms simultaneously.*

## Classification models
### Supervised Learning methods
### Knn(K nearest neighbors)
![Knn](https://user-images.githubusercontent.com/51100947/123881210-e9e87680-d94c-11eb-8785-2286714a45d4.png)

*The algorithms the data spot that we want to predict calculate the neirest spots distance with using euclidean distance and decison the label by neirest spots surrounding a number of labels.*
*The algorithms that use to euclidean distance needs always to normalization because some distances too bigger than the other distances, that will be domination the other distances so it wont't be good perform!!*

### Linear SVM(Support Vector Machine)
![linear_svm](https://user-images.githubusercontent.com/51100947/121974329-d0a9cc80-cd87-11eb-9f9d-fe29e2395f88.png)

*The processes goals optimize best margin by support vectors.*

### Naive Bayes
![naives_bayes](https://user-images.githubusercontent.com/51100947/122481838-1eb50f00-cfd8-11eb-8eed-7282f86ad452.png)

*Naive Bayes algorithm depend of probality by spots position* 

### Decision Tree 
![dt](https://user-images.githubusercontent.com/51100947/123881971-721b4b80-d94e-11eb-85dd-94ea4bb8f065.png)

*Decision Tree try to best splitting for classification, after that using the thresholds that the best splitting when it prediction proceses*

### Random Forest
![Random_Forest_Class](https://user-images.githubusercontent.com/51100947/122482373-1b6e5300-cfd9-11eb-9a51-5e72ff6256a9.png)

*Random Forest has a lot of Decision trees and it use to these Tress for prediction processes.*

## Unsupervised Learning methods
### KMeans
![kmeans](https://user-images.githubusercontent.com/51100947/122658374-1514de00-d175-11eb-93e5-0a90bc7379f3.png)

*It is aimed to decrease the wcss value.*

### Hierarchical Clustering
![Hıerarcial_Clustering](https://user-images.githubusercontent.com/51100947/122804603-c8461a00-d2d0-11eb-8957-990db96a0957.png)

*There is no definitive answer since cluster analysis is essentially an exploratory approach but generally, when specify optimize cluster number we should look Euclidean distances and pick the threshold that distance has longest Euclidean distance.*

### Introduction the Deep Learning with Logistic regression(a separate deep learning repository will be prepared)
*Shortly, deep learning training processes realise with data not a model therefore Deep learning better than machine learning algorithms in big datas.*

![logistic_Regression](https://user-images.githubusercontent.com/51100947/122483410-2b873200-cfdb-11eb-8695-7ff456de6451.png)

*Logistic Regression basic of the neural networks.*

*Forward and backward propagation goals, find the best weigth(w) and bias(b).*

# Encoding Types 
*Machine learning algorithms needs to numerical categorization because the computer needs to understand correlations, so there are two encoding types in sckit-learn library.*

## Label encoder 
*Popular conversion tool for manipulating categorical variables. In this technique, each data is assigned an alphabetical, different integer.*

Basic example for label encoder:

![Label_Encoder](https://user-images.githubusercontent.com/51100947/122485077-bb7aab00-cfde-11eb-9b60-c9bd3880e01e.PNG)

## OneHot encoding
*One Hot Encoding means that categorical variables are displayed as binary (binary). This process must first convert the categorical values to the values of integers. Then its integer value is represented as a binary with all values except the integer index marked with 1.*

Basic example for one hot encoding:

![one_hot_encoding](https://user-images.githubusercontent.com/51100947/122485416-815dd900-cfdf-11eb-94c7-c785e5461297.PNG)

# NLP(Natural Language Process)
*Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.*

## NLP Processes
![nlp_myself](https://user-images.githubusercontent.com/51100947/123517031-b5a05c00-d6a7-11eb-80ac-5e1bd607d104.png)

As we can see, actually the methods has tried to clear texts for computer and has tried to encoding the all letters for models. 

## What is the Bag of words ?
The Bag of Words (BoW) model is the simplest form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).

Let’s recall the three types of movie reviews we saw earlier:

* Review 1: This movie is very scary and long
* Review 2: This movie is not scary and is slow
* Review 3: This movie is spooky and good
*We will first build a vocabulary from all the unique words in the above three reviews. The vocabulary consists of these 11 words: ‘This’, ‘movie’, ‘is’, ‘very’, ‘scary’, ‘and’, ‘long’, ‘not’,  ‘slow’, ‘spooky’,  ‘good’.*

We can now take each of these words and mark their occurrence in the three movie reviews above with 1s and 0s. This will give us 3 vectors for 3 reviews:

![BoWBag-of-Words-model-2](https://user-images.githubusercontent.com/51100947/123517231-9229e100-d6a8-11eb-8c02-37a918e2abea.png)

*Source : https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/

# PCA(Principle Component Analysis)
*Shortly, Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.*

The PCA Explanation with graph

![PCA](https://user-images.githubusercontent.com/51100947/123879158-22865100-d949-11eb-997d-2cb5b00daa86.png)

So what is the variance ?
*As you can see in graph variance tells you the degree of spread in your data set. The more spread the data, the larger the variance is in relation to the mean.*

# What is the K-Fold-CrossValidation
*We can say for K-Fold-CrossValidation, that is testing processes for avoid the overfitting.
How it works ?*

![K-Fold-Cross_Validation](https://user-images.githubusercontent.com/51100947/124141439-1a85f880-da92-11eb-83f7-6f088ffed093.png)

As you can see the process goal split the train data by k number.
In number of 'k-1' in train data split for training process, in number of 1 split in train data for testing process.

## GridSearchCV vs RandomizedCV? 
*In gridsearchCV a process that searches exhaustively through a manually specified subset of the hyperparameter space of the targeted algorithm.*

*In randomizedsearchcv, instead of providing a discrete set of values to explore on each hyperparameter, we provide a statistical distribution or list of hyper parameters. Values for the different hyper parameters are picked up at random from this distribution*
Actually, we can say grid search method important for optimization models processing.

![Grid_Search](https://user-images.githubusercontent.com/51100947/124144153-6043c080-da94-11eb-8c3b-b7f23fca20d7.png)

As we can see diffrences between grid search and random search.
So we can say if we had a time and our model have no complexity we can use gridsearchCV.
But we had a too big data and our model have complexity we can use RandomizedSearchCV.

## What are the Overfitting and Underfitting status ?

*Overfitting basicly, when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. example where given in below for overfitting.*

![image](https://user-images.githubusercontent.com/51100947/125286467-81be6b00-e324-11eb-9e13-99bd04c4912f.png)

* As you can see the overfitting status has some patterns by a train data.

*Underfitting basicly, the counterpart of overfitting, happens when a machine learning model is not complex enough to accurately capture relationships between a dataset’s features and a target variable. Example where given in bellow for underfitting status.*

![underfitting](https://user-images.githubusercontent.com/51100947/125286941-04dfc100-e325-11eb-9def-b286809416c8.PNG)

* As you can see underfitting status has no optimize line for training data.

# Recommendation Systems
*Basicly, a recommendation system , is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.* 

Even if the systems sometimes too dangerous for humanity (as you know that a Cambridge Analytica scandal in USA's presidential election), but it's usefull in someplaces for example like a Netflix, YouTube, Amazon, Facebook application.

How it works the systems ? 
We realize a two types for these systems that are "User based" and "Items based"

![1_QvhetbRjCr1vryTch_2HZQ](https://user-images.githubusercontent.com/51100947/124333367-9ff5cf80-db9c-11eb-9e21-d9d2d72038d6.jpeg)

As you can see user based depends on a users's habits and try to finding similarty with users.
On the other hand items depends on a try to finding similarty with items.
Generally, items based systems more usefull than userbased systems because people sometimes change the their habits so the process won't recommend appropriate product for people in a future, but the items will cannot change in a future and the process optimization might has good performs. 

# Exploraty Data Analysis and training data that we use
### * Audi data
### * Cancer data
### * Stroke Prediction data
### * Biomechanical human data
### * Mall Customers
### * COVID 19 Tweet types
### * Google App reviews

# Audi data
*The data has audi cars 1997 and 2020 between years some features where feaurues in below and trying to guess price in dependent where given in data.*

*Using three regression models because 'price' feature is continuous label*

##### Content
* Data analysis
  * Average price by years
  * Transmision type by years
  * Fuel types by years
  * Model types by years
  * Avarage MPG by years
  * Avarage Engine Size by years
  * General display of numerical features by year
* ML Preprocessing
  * Obtain train and test spliting process
  * Encoding categorical features for learning processes
  * Used to Label encoder.
  * Train and test split
* Learning Time !
  * Linear Regression
    * Learnin curve
  * Decision tree regressor
    * Learnin curve
  * Random Forest regressor
    * Learning curve

# Cancer data
*The data, has cancer cells feature where feature in below and trying to diagnose in dependent of feauture.*
*Using Logistic Regression method the method that are basic of artifical neural networks and good performs if data has a binary labels.*

##### Content
* ML preprocessings
* Tryin to understand data
* Encoding labels with LabelEncoder method
* Implement to variables
* Normalization values
* Train and test splitting
* Learning Time!
  * Logistic Regression
     * Implementing initilazing parameters and sigmoid function
     * Implementing forward and backward propagation
     * Implementing and update parameters
     * Implementing prediction
     * Implementing Logistic Regression
     * Logistic Regression with sklearn
  * Grid searching for best hyperparameters
     * Knn classification
     * Linear SVM
     * Decision Tree Classification 
     * Random Forest Classification
     * Naive Bayes Classification
* Compare the learning algorithms
   * Visualization part
   * Confusion matrixes

# Stroke Prediction data 

*The stroke prediction data set trying to guess to stroke status in dependent some features where given in data.*

*The data is also imbalanced data, it was good experiences for me.*

##### Content
* Imported data
* Load to data
* Tried to understand data
* Control the missing value
* Feature engineering
  * Correlation numerical values
  * Smoking status by gender type and relation with stroke
  * Smoking status by work type relation with stroke
  * Smoking status by age mean relation with stroke
  * Smoking status by avg_glucose_level mean relation with stroke
  * Smoking status by bmi mean relation with stroke
  * Genaral visualization for that we did feature engineering
  * Density map of numerical values(hypertension; age and bmi level relation)
* Examined to median values in numerical values
* Control the outlier values for numerical values
* Filling the missing value
* Obtained training and testing variables for learning proceses
* Encoding catagorical features with label encoder for learning processes
* Train and test splitting and try to balanced labels
* Normalization for continous columns
* Implement the PCA(Principle Component Analysis) data and 2D visualization
* Learning time!
  * Optimization hyperparameters with RandomizedSearchCV
    * Logistic Regression
       * Fitting and testing model
    * Knn Classification
       * Fitting and testing model
    * Decision Tree Classification
       * Fitting and testing model

# Biomechanical human data

*This content has Biomechanical Feautures of orthopedic patiens dataset. Humans's bimechanicals has a a lot of feaure that you can see features below.*

##### Content
* Load to data
* Trying to understand data
* Feature engineering
  * Analyze the correlation between features
  * lumbar_lordosis_angle and pelvic_incidence
  * degree_spondylolisthesis and pelvic_incidence
  * pelvic_tilt_number and pelvic_incidence
  * lumbar_lordosis_angle and sacral_slope
* Train and test splitting processes
  * Decleration variabels for splitting processes
  * Encoding labels(object to int64) with label encoding method
  * Test and train splitting
* Normalization for numerical values
* Learning time!
  * Logistic Regression
  * Knn classification
  * Linear SVM
  * Decision Tree Classification 
  * Random Forest Classification
  * Naive Bayes Classification
* Compare the learning algorithms
   * Visualization part
   * Confusion matrixes


# Mall Customers

*The dataset has mall customers calculate the spending score with features where given in data.*
* The data has no label and it is avalaible for clustering.
* We will cluster the datas with Kmeans algorithms(Unsupervised method).

##### Content
* Trying to understand data
* Encoding feature that has a object type
* Feature engineering
  * Analysis correlation values.
  * Avarage to anual income by gender.
  * Avarage to spending score by gender.
  * Relation with age and spending score.
    * General
    * By gender
  * Relation with age and Anual incomes
    * General
    * By gender
  * Relation with Anual incomes and Spending score
    * General
    * By gender
* Dropping to 3 feature for clustering and visualization
* Clustering time!
* KMeans
  * Specify k number with elbow method.
  * Clustering with kmeans algorithms
  * Visualization Clusters and centroids.
* Hierarcial Clustering(HC)
  * Visualization Dendogram map and decide to cluster number
  * Implement the HC algorithms
  * Visualization Clusters.

# COVID-19 Tweet types
### Introduction
The data set has tweet that relation with Covid - 19, also has a label for tweet type that

The content predict the positive, negative, neutral, extremely positive, extremely Negative tweets type with NLP.

Content
* Importing libraries
* Load to data
* Trying to understand data
* Visualization to percantage tweet types general
* Analyzing the tweets type location that more than has 100 tweets
* Detecthing location that has more than 100 tweets
   * Create new data frame for tweets type that has more than 100 tweets
   * Filling the data frame that we created
   * Visualization to percantage tweets type for Covid 19 by location that has more than 100 tweets
* NLP(Natural Language Processing) Processes
   * Pick the indices for testing preprocessing processes
   * Remove the ırrevelant strings(: , :) , ! , //...) and convert the lower case
   * Tokinenize to text
   * Lemmization all words, and convert again text form
* Create a new data frame for tweet types and tweets
* Data cleaning
  * Dropping NaN values
  * The processes that we testing on one indices implement on data.
  * Remove the stopwords and implement the countvectorizer
  * Bag of words
* Encoding labels with LabelEncoder method
* The most use words, with data visiulazation
* Splitting to train and test data
* Learning Time!
  * Logistic Regression
  * Confusion matrix

# Google Appstore reviews

### Introduction
The data set GoogleAppStore reviews for mobile application, and also label for positive, negative and neutral label.

The content predict the positive, negative and neutral tags of the reviews with NLP.

Content
* Importing libraries.
* Trying to understand data.
* Create a new data frame for replies and labels(Concat feature).
* Dropping NaN replies from data frame.
* Pick the indices for testing preprocessing processes
  * Remove the redundant strings(these are ':' , ':)' , '//'...).
  * Split to words.
  * Lemmazation.
* The processes that we testing on one indices implement on data.
  * Bag of words.
  * Create the 'Sparce Matrix' for bag of words.
* The most use words, with data visiulazation
* Learning Time!
  * Split the train and test data.
  * Training with Random Forest.
  * Training with Logistic Regression
* Confusion matrixes
 * Random Forest
 * Logistic Regression
