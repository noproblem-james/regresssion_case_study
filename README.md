# Sale Price of Heavy Equipment at Auction
## Context
This was a competition among several three-person teams performed in "hackathon" style over the course of six hours. I became fascinated with the problem and the dataset, so scrapped the work I did with my team, started over, and took it much further. This was one of the first end-to-end data science projects I completed, and I learned a lot of valuable lessons, some of which I've included at the bottom of this readme, as a reminder of how far I've come.

## The Objective

The goal of the competition was to predict the sale price of a particular piece of heavy equipment at auction based on its usage, equipment type, and configuration.  The data come from auction result postings.


* The data are quite messy. In particular, there are lots of categorical variables with lots of missing values.

* Contestants were originally confined to linear models.

* Models were evaluated using Root Mean Squared Logarithmic Error (RMSLE).

The (RMSLE) Formula looks like this:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the target values.

This loss function is sensitive to the *ratio* of predicted values to the actual values, a prediction of 200 for an actual value of 100 contributes approximately the same amount to the loss as a prediction of 2000 for an actual value of 1000.  This is because  a difference of logarithms is equal to a single logarithm of a ratio.


## Approach:  
* EDA
* First pass for model fitting:
  * Age
    * Convert "saledate" column to datetime format
    * Break out datetime into different components (year, month, day of month, day of week)
   * Munge `YearMade` to strip out low vals
   * Product Group
   * Product Size

* Second pass -- Explore these options:
   * Product class (contains information about "heft" of machinery... how much power it has, how deep it digs, etc.)
   * Data Source
   * AuctioneerID
   * State - Large # of Dummies -- Group into regions?
   * Enclosure

* Compare linear models to ensemble tree models


## Lessons:


1. **Start simply:**
  * Triage. Use judgement about where your cleaning efforts will yield the most results and focus there first.
  * Try to fit a basic model within an hour or two, even knowing it has some weaknesses

2. **Use intuition.**
 - Think about which features may send the strongest signal, as if you weren't fitting a model.
 - This can be done *without looking at the data*; thinking about the problem has no risk of overfitting.

3. **Avoid data leakage**
  - Any transformations of the training data that *learn parameters* (e.g., standardization learns the mean and variance of a feature) must only use parameters learned from the *training data*.

4. **Linear models have limitations.**
  - In addition to making theoretical assumptions for he purpose of statistical inference, linear models pose some practical problems..
  - Must consider how to normalize continuous predictors.
  - Must create lots of dummy variables for categorical predictors.
    - Creating an ordinal variable (e.g., converting the values `Small`, `Medium` and `Large` to the numbers `1`, `2`, and `3`, respectively) will lock you into a linear form, when that may not be the true relationship. For instance, it's possible that a response variable like `price`, is higher for `Medium` than for `Small`, but is actually *lower* for `Large`. This isn't a problem for tree-based methods, which makes no assumptions about linear relationships.
    - Creating dummy variables means adding a lot of columns in a consistent way to both training and evaluation sets. This becomes problematic, because some columns in the test data will take on values not seen in the training data, and vice versa.
  - Without workarouds, a linear model can produce negative predictions, even when that is nonsensical (as is the case with a sale price).


5. **Mini Lessons:**
  - Use pd.merge() instead of pd.join() in most instances
  - Look for encoding of strings by printing an individual cell when things don't match up correctly.
  - Need special procedure to get matplotlib to plot datetime values
  - Need an entirely new approach to avoid the common pandas problem  when trying to map new values onto a column or create a new column
    ```python
    "A value is trying to be set on a copy of a slice from a DataFrame"
    ```
    - Sometimes it does this for .replace(), .dropna(), new column creation (df["newcol"] = df[col1] + df[col2]). even when using .loc[], as suggested by docs.


## TODO:
* Remove SalesID from train and test data.
* Add in `yearmade`
* Add higher-order terms to linear models
* Fit a ridge regressor
* Partition data by Product Group, build several (linear) models, one for each group, and include all machine configuration variables when fitting.
* Find a better way to deal with negative age values, replace sale date and year made with median values for that model number
* Find a better way to dummify data, using dicts rather than the getdummies() built-in function.
* Implement a preprocessing pipeline and a gridsearch for ensemble methods.
* Fit XG-Boost regressor
* How/why could Lasso model perform worse than non-regularized linear model, out of sample?
* How/why could ensemble models perform worse, out of sample (especially random forest, which resists overfitting)? Is it not extrapolating well because test data varies quite a bit from train data?
  * regression r^2 : .65
  * random forest r^2: .90
* Why is k-folds cross-validation failing for linear regression when k > 3? (Still works for ensemble methods)
