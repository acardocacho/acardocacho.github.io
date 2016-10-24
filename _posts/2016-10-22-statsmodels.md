---
layout: post
title: 2016-10-22-statsmodels.md
---

# Linear Regression with Statsmodels and Scikit-Learn

Let's investigate the housing dataset with linear regression. Here's the documentation for `statsmodels` (in case you need it):
* statsmodels -- [linear regression](http://statsmodels.sourceforge.net/devel/examples/#regression)

## Intro to Statsmodels

Statsmodels is a python package that provides access to many useful statistical calculations and models such as linear regression. It has some advantages over `scikit-learn`, in particular easier access to various statistical aspects of linear regression.

First let's load and explore our dataset, then we'll see how to use statsmodels. We'll use `sklearn` to provide the data.


```python
%matplotlib inline
from matplotlib import pyplot as plt

from sklearn import datasets
data = datasets.load_boston()

print data.DESCR
```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    


Let's take a minute to see what the data looks like.


```python
print data.feature_names
print data.data[0]
print data.target[0]
```

    ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
     'B' 'LSTAT']
    [  6.32000000e-03   1.80000000e+01   2.31000000e+00   0.00000000e+00
       5.38000000e-01   6.57500000e+00   6.52000000e+01   4.09000000e+00
       1.00000000e+00   2.96000000e+02   1.53000000e+01   3.96900000e+02
       4.98000000e+00]
    24.0


Scikit-learn has already split off the house value data into the target variable. Let's see how to build a linear regression. First let's put the data into a data frame for convenience, and do a quick check to see that everything loaded correctly.


```python
type(data)
```




    sklearn.datasets.base.Bunch




```python
type(data.data)
```




    numpy.ndarray




```python
type(data.feature_names)
```




    numpy.ndarray




```python
type(data.target)
```




    numpy.ndarray




```python
type(data.data[0])
```




    numpy.ndarray




```python
import numpy as np
import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
targets = pd.DataFrame(data.target, columns=["MEDV"])

# Take a look at the first few rows
print df.head()
print targets.head()
```

          CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
    0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
    1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
    2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
    3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
    4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
    
       PTRATIO       B  LSTAT  
    0     15.3  396.90   4.98  
    1     17.8  396.90   9.14  
    2     17.8  392.83   4.03  
    3     18.7  394.63   2.94  
    4     18.7  396.90   5.33  
       MEDV
    0  24.0
    1  21.6
    2  34.7
    3  33.4
    4  36.2


Now let's fit a linear model to the data. First let's take a look at some of the variables we identified visually as being linked to house value, RM and LSTAT. Let's look at each individually and then both together.

Note that statsmodels does not add a constant term by default, so you need to use `X = sm.add_constant(X)` if you want a constant term.


```python
import statsmodels.api as sm

X = df["RM"]
y = targets["MEDV"]

# Note the difference in argument order
# ordinary least squares
model = sm.OLS(y, X).fit()     # build and train the model
predictions = model.predict(X) # make predictions

# Print out the statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th> <td>   0.901</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.901</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4615.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 24 Oct 2016</td> <th>  Prob (F-statistic):</th> <td>3.74e-256</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:52:21</td>     <th>  Log-Likelihood:    </th> <td> -1747.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3496.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   505</td>      <th>  BIC:               </th> <td>   3500.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>RM</th> <td>    3.6534</td> <td>    0.054</td> <td>   67.930</td> <td> 0.000</td> <td>    3.548     3.759</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>83.295</td> <th>  Durbin-Watson:     </th> <td>   0.493</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 152.507</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.955</td> <th>  Prob(JB):          </th> <td>7.65e-34</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.894</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table>



### Interpreting the Coefficients

Here the coefficient of 3.634 means that as the `RM` variable increases by 1, the predicted value of `MDEV` increases by 3.634.

Let's plot the predictions versus the actual values.


```python
# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM")
plt.ylabel("Actual Values MEDV")
plt.show()
print "MSE:", model.mse_model
```


![png](../images/2016-10-22-notebook/output_16_0.png)


    MSE: 20654.4162188


**Check**: How does this plot relate to the model? In other words, how are the independent variable (RM) and dependent variable ("MEDV") incorporated?

Solution: They are used to make the predicted values (the x-axis)

Let's try it with a constant term now.


```python
## With a constant

import statsmodels.api as sm

X = df["RM"]
X = sm.add_constant(X)
y = targets["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th> <td>   0.484</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.483</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   471.8</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 24 Oct 2016</td> <th>  Prob (F-statistic):</th> <td>2.49e-74</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:03:11</td>     <th>  Log-Likelihood:    </th> <td> -1673.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3350.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3359.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th> <td>  -34.6706</td> <td>    2.650</td> <td>  -13.084</td> <td> 0.000</td> <td>  -39.877   -29.465</td>
</tr>
<tr>
  <th>RM</th>    <td>    9.1021</td> <td>    0.419</td> <td>   21.722</td> <td> 0.000</td> <td>    8.279     9.925</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>102.585</td> <th>  Durbin-Watson:     </th> <td>   0.684</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 612.449</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.726</td>  <th>  Prob(JB):          </th> <td>1.02e-133</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.190</td>  <th>  Cond. No.          </th> <td>    58.4</td> 
</tr>
</table>




```python
# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM")
plt.ylabel("Actual Values MEDV")
plt.show()
print "MSE:", model.mse_model
```


![png](../images/2016-10-22-notebook/output_19_0.png)


    MSE: 20654.4162188


### Interpreting the Coefficients

With the constant term the coefficients are different. Without a constant we are forcing our model to go through the origin, but now we have a y-intercept at -34.67. We also changed the slope of the `RM` regressor from 3.634 to 9.1021.

Next let's try a different predictor, `LSTAT`.



```python
X = df[["LSTAT"]]
y = targets["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th> <td>   0.449</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.448</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   410.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 24 Oct 2016</td> <th>  Prob (F-statistic):</th> <td>2.71e-67</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:04:42</td>     <th>  Log-Likelihood:    </th> <td> -2182.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   4367.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   505</td>      <th>  BIC:               </th> <td>   4371.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>LSTAT</th> <td>    1.1221</td> <td>    0.055</td> <td>   20.271</td> <td> 0.000</td> <td>    1.013     1.231</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.113</td> <th>  Durbin-Watson:     </th> <td>   0.369</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.573</td> <th>  Jarque-Bera (JB):  </th> <td>   1.051</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.112</td> <th>  Prob(JB):          </th> <td>   0.591</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.009</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table>




```python
# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from LSTAT")
plt.ylabel("Actual Values MEDV")
plt.show()
print "MSE:", model.mse_model
```


![png](../images/2016-10-22-notebook/output_22_0.png)


    MSE: 134427.133958


Finally, let's fit a model using both `RM` and `LSTAT`.


```python
X = df[["RM", "LSTAT"]]
y = targets["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th> <td>   0.948</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.948</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4637.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 24 Oct 2016</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>14:11:52</td>     <th>  Log-Likelihood:    </th> <td> -1582.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3170.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3178.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>RM</th>    <td>    4.9069</td> <td>    0.070</td> <td>   69.906</td> <td> 0.000</td> <td>    4.769     5.045</td>
</tr>
<tr>
  <th>LSTAT</th> <td>   -0.6557</td> <td>    0.031</td> <td>  -21.458</td> <td> 0.000</td> <td>   -0.716    -0.596</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>145.153</td> <th>  Durbin-Watson:     </th> <td>   0.834</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 442.157</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.351</td>  <th>  Prob(JB):          </th> <td>9.70e-97</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.698</td>  <th>  Cond. No.          </th> <td>    4.72</td>
</tr>
</table>




```python
# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM and LSTAT")
plt.ylabel("Actual Values MEDV")
plt.show()
print "MSE:", model.mse_model
```


![png](../images/2016-10-22-notebook/output_25_0.png)


    MSE: 142090.70278


## Comparing the models

A perfect fit would yield a straight line when we plot the predicted values versus the true values. We'll quantify the goodness of fit soon.

### Exercise

Run the fit on all the variables with `X = df`. Did this improve the fit versus the previously tested variable combinations? (Use mean squared error).


```python

```

## Preparing data with Patsy

`Patsy` is a python package that makes preparing data a bit easier. It uses a special formula syntax to create the `X` and `y` matrices we use to fit our models with.

Let's look at a few examples. To get the `X` and `y` matrices for the previous example, try the following.


```python
import patsy

# First let's add the targets to our data frame
df["MEDV"] = targets["MEDV"]

y, X = patsy.dmatrices("MEDV ~ RM + LSTAT", data=df)
print X[0:5, :]
print y[0:5, :]
```

    [[ 1.     6.575  4.98 ]
     [ 1.     6.421  9.14 ]
     [ 1.     7.185  4.03 ]
     [ 1.     6.998  2.94 ]
     [ 1.     7.147  5.33 ]]
    [[ 24. ]
     [ 21.6]
     [ 34.7]
     [ 33.4]
     [ 36.2]]


We can also apply functions to our data in the formula. For example, to perform a quadratic regression of "MEDV" with "LSTAT", we would use the following formula.


```python
y, X = patsy.dmatrices("MEDV ~ LSTAT + I(LSTAT**2)", data=df)
print X[0:5, :]
print y[0:5, :]
```

    [[  1.       4.98    24.8004]
     [  1.       9.14    83.5396]
     [  1.       4.03    16.2409]
     [  1.       2.94     8.6436]
     [  1.       5.33    28.4089]]
    [[ 24. ]
     [ 21.6]
     [ 34.7]
     [ 33.4]
     [ 36.2]]


You can use some python functions, like `numpy`'s power.


```python
y, X = patsy.dmatrices("MEDV ~ LSTAT + np.power(LSTAT,2)", data=df)
print X[0:5, :]
print y[0:5, :]
```

    [[  1.       4.98    24.8004]
     [  1.       9.14    83.5396]
     [  1.       4.03    16.2409]
     [  1.       2.94     8.6436]
     [  1.       5.33    28.4089]]
    [[ 24. ]
     [ 21.6]
     [ 34.7]
     [ 33.4]
     [ 36.2]]


Patsy can also handle categorical variables and make dummy variables for you.


```python
from patsy import dmatrix, demo_data

data = demo_data("a", nlevels=4)
print data
dmatrix("a", data)
```

    {'a': ['a1', 'a2', 'a3', 'a4', 'a1', 'a2', 'a3', 'a4']}





    DesignMatrix with shape (8, 4)
      Intercept  a[T.a2]  a[T.a3]  a[T.a4]
              1        0        0        0
              1        1        0        0
              1        0        1        0
              1        0        0        1
              1        0        0        0
              1        1        0        0
              1        0        1        0
              1        0        0        1
      Terms:
        'Intercept' (column 0)
        'a' (columns 1:4)



## Guided Practice

### Exercises

Practice using patsy formulas and fit models for
* CRIM and INDUS versus MDEV (price)
* AGE and CHAS (categorical) versus MDEV


```python
# CRIM and INDUS versus MDEV
y, X = patsy.dmatrices("MEDV ~ CRIM + INDUS", data=df)
print X[0:5, :]
print y[0:5, :]
```

    [[  1.00000000e+00   6.32000000e-03   2.31000000e+00]
     [  1.00000000e+00   2.73100000e-02   7.07000000e+00]
     [  1.00000000e+00   2.72900000e-02   7.07000000e+00]
     [  1.00000000e+00   3.23700000e-02   2.18000000e+00]
     [  1.00000000e+00   6.90500000e-02   2.18000000e+00]]
    [[ 24. ]
     [ 21.6]
     [ 34.7]
     [ 33.4]
     [ 36.2]]



```python
# AGE and CHAS versus MDEV 
# CHAS is categorical, needs dummy variables.
# Actually, it is already a binary value (see dataset description)
y, X = patsy.dmatrices("MEDV ~ AGE + CHAS", data=df)
print X[0:5, :]
print y[0:5, :]
```

    [[  1.   65.2   0. ]
     [  1.   78.9   0. ]
     [  1.   61.1   0. ]
     [  1.   45.8   0. ]
     [  1.   54.2   0. ]]
    [[ 24. ]
     [ 21.6]
     [ 34.7]
     [ 33.4]
     [ 36.2]]


## Independent Practice

Try to find the best models that you can that:
* use only two variables
* only three variables
* only four variables

Evaluate your models using the squared error. Which has the lowest? How do the errors compare to using all the variables?

### Exercise

From the LSTAT plot you may have noticed that the relationship is not quite linear. Add a new column `"LSTAT2"` to your data frame for the LSTAT values squared and try to fit a quadratic function using `["LSTAT", "LSTAT2"]`. Is the resulting fit better or worse?


```python

```

## Bonus

We'll go over using Scikit-Learn later this week, but you can get a head start now by repeating some of the exercises using `sklearn` instead of `statsmodels`.

### Exercises

Recreate the model fits above with `scikit-learn`:
* a model using LSTAT
* a model using RM and LSTAT
* a model using all the variables

Compare the mean squared errors for each model between the two packages. Do they differ significantly? Why or why not?


```python

```
