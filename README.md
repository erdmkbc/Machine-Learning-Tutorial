# Machine Learning 
What is the Machine Learning ?

*Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.*




# Content
### * Audi data
### * Cancer data
### * Stroke Prediction data
### * Biomechanical human data
### * The machine learning algorithms that we used

## It contains 4 data for now

### Audi data
*The data has audi cars 1997 and 2020 between years some features where feaurues in below and trying to guess price in dependent where feature in below.*

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

### Cancer data
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

### Stroke Prediction data 

*The stroke prediction data set trying to guess to stroke status in dependent some features where features below.*

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


### Biomechanical human data

*This content has Biomechanical Feautures of orthopedic patiens dataset. Humans's bimechanicals has a a lot of feaure that you can see features below.*

The training processes was implemented with,

* Logistic Regression
* Knn
* Linear SVM methods.

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

# The machine learning algorithms that we used

## Linear Regression
![linear reggreasion](https://user-images.githubusercontent.com/51100947/121821481-01afd180-cca2-11eb-929f-32bf14b587ef.png)

The model usefull for continous labels 

*The goal is try to draw the most optimized line.*
*Tryin to minimize MSE(mean squared eror).*

![polynomial regression](https://user-images.githubusercontent.com/51100947/121821640-f01af980-cca2-11eb-97cd-3d114ddf41f8.png)
#### *If our data has parabolic density we can use polynomial form.*


## Decision Tree 
![decision_tree](https://user-images.githubusercontent.com/51100947/121821686-2789a600-cca3-11eb-9fb0-2f8e7d654095.png)

*Algorithm that the best try to splitting the coordinate plane into many parts(terminal leaf we can say) and makes predictions as a result of comparisons.*

## Random forest 
![randon_forest](https://user-images.githubusercontent.com/51100947/121822495-0bd4ce80-cca8-11eb-944a-e897711e047d.png)

*The algorithm usefull for recommendation algorithms(for example Netflix and YouTube recommendations)*
*Random forest actually, has a lot of decision trees's results that we avarage results.*
*Ensemble learning family that using multiple ml algorithms simultaneously.*

## Logistic Regression 
![logistic_regression](https://user-images.githubusercontent.com/51100947/121822774-894d0e80-cca9-11eb-9d07-14bd36e64128.png)

*The algoritms usefull for data that has a binary label.* 

*Logistic Regression is basic of the neural networks.*

*The features are need to numerical values.*

*Learning procceses that aim to optimize the w and bias values by continuously implementing forward and backward propagation.*

## Knn(K nearest neighbors)
![knn](https://user-images.githubusercontent.com/51100947/121974267-b2dc6780-cd87-11eb-9b9d-1e16862cf7d3.png)

## Linear SVM(Support Vector Machine)
![linear_svm](https://user-images.githubusercontent.com/51100947/121974329-d0a9cc80-cd87-11eb-9f9d-fe29e2395f88.png)

*The processes goals optimize best margin by support vectors.*
