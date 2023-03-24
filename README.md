# power outage prediction
by Tongxun Hu and Jessica Zhang

# Framing the Problem

For this power outage dataset, we try to predict the cause of a major power outage. We are trying to make prediction for the major power outages after 2016, so that we assume that we can access the entire dataframe of power outage (range from 2000 to 2016) to train our model. We will use the already cleaned data from project 3. The response variable will be the CAUSE.CATEGORY column in the dataset; we choose it because we think that it is closely related to the power outage, so it will predict the power outage in an effective way. This prediction will be a classification and we will use multi-class classification since there are multiple causes of power outages. We choose to use accuracy to evaluate our model because it measures the overall performance of the prediction based on the entire set of data. The dataset doesn't have class imbalance; each cause weighs the same and the errors are equally bad, so accuracy can be used to make sure that they contribute equally when calculating the accuracy value. 


------
# Baseline Model

For the baseline model, we used CLIMATE.CATEGORY, MONTH, and OUTAGE.DURATION(mins) as features to predit the cause of power outages. CLIMATE.CATEGORY is the climate episodes corresponding to the years. The categories—“Warm”, “Cold” or “Normal” episodes of the climate are based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI), and it's nominal data. MONTH is the month that a power outage happened, which is also a nominal data. OUTAGE.DURATION(mins) is the duration of a power outage, which is a quantitative data. 

We first used ColumnTransformer to manipulate these three columns by onehotencoding the CLIMATE.CATEGORY and MONTH column (since they are categorical variables), and standardizing the OUTAGE.DURATION(mins) column so that we could compare the duration time more effectively and in a meaningful way. We would drop other columns in the dataframe by setting the remainder to drop. Then, we put all these transformations into the pipeline and used LogisticRegression to train our model. 

We drop rows that have missing values in any of the columns 'MONTH', 'CLIMATE.CATEGORY', 'OUTAGE.DURATION(mins)'. By looking at our dataset, we observed that 'OUTAGE.DURATION(mins)' is missing if and only if 'OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)' is also missing. Therefore, we assume that the designers of the data collection process intentionally decided to not have data in 'OUTAGE.DURATION(mins)' column when 'OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)' is missing. Thus, the values in 'OUTAGE.DURATION(mins)' column are missing by design. Since the missing values in 'OUTAGE.DURATION(mins)' column are not a lot, only around 3.7%, dropping these values may not significantly introduce bias or affect the overall quality and representativeness of the data. Also, it may not significantly affect the results of the analysis. In this case, it may be acceptable to exclude the missing values in 'OUTAGE.DURATION(mins)' column from our analysis. Moreover, there are 9 missing values in both the 'MONTH' and 'CLIMATE.CATEGORY' columns, located in the same rows. Most of the other data in these rows is also missing. In our case, we may assume that dropping these missing values in 'MONTH' and 'CLIMATE.CATEGORY' columns will not significantly change the data. The response variable column didn't contain nan, so we didn't need to perform any manipulation.

Lastly, we fit the pipeline to the training data and evaluated the accuracy of both training and testing data. After splitting the dataset into training and testing data, the performance, or the model’s ability to generalize to unseen data, was around 0.6 for the testing data, and 0.63 for the training data. 

We believe that the accuracy of the current model is reasonable, but not very good, since we have only considered three features in our model, while in reality, the cause of a power outage is related to many different factors. In addition, the strength of correlations between different columns are not very strong after exploring dataset by drawing various graphs in project 3. 


------
# Final Model

For the final model, we added U.S._STATE and CUSTOMERS.AFFECTED as our new features. We belive that U.S._STATE is good for prediction because each state has different factors such as environments and policies that might affect the cause of power outage differently. CUSTOMERS.AFFECTED is likely related to the cause since certain causes might lead to more numbers of customers affected, and vice versa. 

We first dropped all nan values in the newly added feature columns. Before we created column transformers, we first defined the class StdScalerByGroup that we created in lab 9 so that we can starndardize values based on groups. In the column transformers, in addition to the baseline model, we onehotencode the U.S._STATE, and used the parameter handle_unknown = 'ignore' to avoid error when a category is not found in the training dataset. We also used StdScalerByGroup to standardize CUSTOMERS.AFFECTED based on CLIMATE.CATEGORY. Lastly, we would drop other columns in the dataframe by setting the remainder to drop. 

We first chose logistic regression as our model. Since we didn't talk much about hyperparameters of logistic regression, we did an online search and found that the main hyperparameters we tune here are solver and penalty. We get different results while tuning the hyperparameters. We create alist of lists to keep track of possible solver and penalty combinations. After we used for loop trying to loop over all combinations and ran it several times, we found that the model with penalty=l1 and solver='sage' shows a better performance. The solver algorithm is used for optimizing the problem and can be selected from a list of choices including 'newton-cg', 'lbfgs', 'liblinear', 'sag', and 'saga', with the default being 'lbfgs'. Penalty, is used to prevent overfitting by discouraging the model from becoming too complex. Available penalty options include 'l1', 'l2', 'elasticnet', and 'none', with the default being 'l2'. However, it's possible that some penalties may not work with certain solvers. 

Then we tried to use DecisionTreeClassifier as the model. The hyperparameters that we chose from are: 
`hyperparameters = {'max_depth': [2, 3, 4, 5, 7, 10, 13, 15, 18, None], 'min_samples_split': [2, 5, 10, 20, 50, 100, 200],'criterion': ['gini', 'entropy']}` Then we used GridSearchCV to find the most optimal hyperparameters, which are: 'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 5. 

We didn't gain any substantial difference. However, DecisionTreeClassifier had a slightly better performance than the logistic regression in predicting unseen data. For the DecisionTreeClassifier, the accuracy for the training data was around 0.83, and about 0.82 for the testing data. Therefore, the accuracy was improved significantly compared to the baseline model (0.6).


------
# Fairness Analysis



