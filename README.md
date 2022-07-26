
# House Price Prediction 🏘️

In this project, we will build a linear regression machine learning model  on the Boston housing dataset. This model will help us make better real estate decisions by providing house price predictions for that area.


## Project Flow 📈

![](image.jpg)
## How to run the code?🏃‍♂️
1. Used Visual Studio Code or Google Colab to execute this project.
2. Ensure that all the required packages been installed.
3. Download the ipynb and store it on jupyter directory.
4. ipynb can run each cell at time or can be run completely in on go.
## Reference🔗

[Loading the dataset]('https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html')

[Boston Housing Dataset]('https://github.com/selva86/datasets/blob/master/BostonHousing.csv')

[What is Linear Regression?]('https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/introduction-to-trend-lines/a/linear-regression-review')
## Requriements🔨
*This project is written in python, so the required python packages are give below:*

- For importing and visualizing Dataset following packages are needed
    
        import pandas as pd  
        import numpy as np
        import matplotlib.pyplot as plt 
        %matplotlib inline
        import seaborn as sns 

- For building a machine learing model

        from sklearn.linear_model import LinearRegression
        
- For model testing
        
        from sklearn import metrics
## Explanation of the code🔎

#### Importing libraries and dataset
In this section we will loading all the required libraries which we will need to develop, visualize and test
our model. We will also be loading our dataset from Sklearn.

- Loading the libraries

        import numpy as np
        import matplotlib.pyplot as plt 
        from sklearn import metrics
        import pandas as pd  
        import seaborn as sns 
        from scipy import stats
        from sklearn.datasets import load_boston
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics
        from sklearn import model_selection
        import math
        %matplotlib inline
- Loading Dataset

        boston_dataset = load_boston()
        boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

#### Data exploration and preprocessing
In this section we will analyse our dataset using different methods and then we'll create a
dataframe using the same. We will also carry out preprocessing on the dataframe for using
the linear regression model.

#### Model Implementation
In this section we will import linear regression model from Sklearn. Use features identified
from heatmap and label to create training and testing set. Finally we will train our model
using training set.

- Select features for creating training and test set.

        x1 = bdf[['NOX','RM','DIS','PTRATIO','LSTAT' ]]
        y1 = bdf['MEDV']
- Use train_test split() from Sklearn to create train and test sets

        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33,random_state = 5 )
- Use regression on training data

        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df

#### Model Testing
In this section we will test our prediction with testing data and calculate R2 score to
measure model accuracy. We will also plot the results of the linear regression model.

- Display the following: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R2 Score.

        metrics.mean_absolute_error()
        metrics.mean_squared_error()
        np.sqrt(metrics.mean_squared_error())
        metrics.r2_score()

## Results📝

You should be able to achieve the following:
- Display and interpret the R2 score and results of error functions
- Create and display a plot using negative slope and positive y intercept obtained from polyfit() function for Median value label and Nitric oxide content attribute.
- Create and display a pairplot of the attributes against the label where attributes with positive corelation with the label will have plots with positive slope and attributes having negative corelation will have negative slope.
