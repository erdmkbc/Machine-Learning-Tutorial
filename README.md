# Machine Learning Tutorial

*This repository tutorial for Machine learning preprocesses, Exploraty Data Analysis, Machine learning algorithms. I hope it will be helpful :).*

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
     * Multiple linear regression
     * Polynomial linear regression
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
![linear_regression](https://user-images.githubusercontent.com/51100947/122485607-08ab4c80-cfe0-11eb-80e9-b4e95c7c5609.png)

The model usefull for continous labels 

*The goal is try to draw the most optimized line.*
*Tryin to minimize MSE(mean squared eror).*

### Polynomial Regression
![polynomial regression](https://user-images.githubusercontent.com/51100947/121821640-f01af980-cca2-11eb-97cd-3d114ddf41f8.png)

*If our data has parabolic density we can use polynomial form.*

### Decision Tree 
![decision_tree](https://user-images.githubusercontent.com/51100947/121821686-2789a600-cca3-11eb-9fb0-2f8e7d654095.png)

*Algorithm that the best try to splitting the coordinate plane into many parts(terminal leaf we can say) and makes predictions as a result of comparisons.*

### Random forest 
![randon_forest](https://user-images.githubusercontent.com/51100947/121822495-0bd4ce80-cca8-11eb-944a-e897711e047d.png)

*The algorithm usefull for recommendation algorithms(for example Netflix and YouTube recommendations)*
*Random forest actually, has a lot of decision trees's results that we avarage results.*
*Ensemble learning family that using multiple ml algorithms simultaneously.*

## Classification models
### Supervised Learning methods
### Knn(K nearest neighbors)
![knn](https://user-images.githubusercontent.com/51100947/121974267-b2dc6780-cd87-11eb-9b9d-1e16862cf7d3.png)

### Linear SVM(Support Vector Machine)
![linear_svm](https://user-images.githubusercontent.com/51100947/121974329-d0a9cc80-cd87-11eb-9f9d-fe29e2395f88.png)

*The processes goals optimize best margin by support vectors.*

### Naive Bayes
![naives_bayes](https://user-images.githubusercontent.com/51100947/122481838-1eb50f00-cfd8-11eb-8eed-7282f86ad452.png)

*Naive Bayes algorithm depend of probality by spots position* 

### Decision Tree 
![DecisionTree_Class](https://user-images.githubusercontent.com/51100947/122482137-aa2ea000-cfd8-11eb-8e41-1212fdbb5a7f.png)

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

# Exploraty Data Analysis and training data that we use
### * Audi data
### * Cancer data
### * Stroke Prediction data
### * Biomechanical human data
### * Mall Customers
## It contains 5 data for now

# Audi data
*The data has audi cars 1997 and 2020 between years some features where feaurues in below and trying to guess price in dependent where given in data.*

*Using three regression models because 'price' feature is continuous label*

##### Content
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
* Decision tree regressor
* Random Forest regressor
* Compare the regressions's r2 score

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

*Using Logistic Regression method the method that are basic of artifical neural networks and good performs if data has a binary labels.*

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

