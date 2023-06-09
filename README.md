# The Cause of Major Power Outage
by Tongxun Hu and Jessica Zhang

# Framing the Problem

For this power outage dataset, we try to predict the cause of a major power outage. We are trying to make prediction for the cause of major power outages after the power has been restored, so we assume that we can access all the necessary columns of power outage dataset, which includes columns such as duration, customers affected, etc. to train our model. We will use the already cleaned data from project 3. The response variable will be the CAUSE.CATEGORY column in the dataset; we choose it because we think that it is closely related to the power outage, so it will predict the power outage in an effective way. This prediction will be a classification and we will use multi-class classification since there are multiple causes of power outages. 

In order to deal with missing data, we dropped rows that have missing values in any of the columns 'MONTH', 'CLIMATE.CATEGORY', 'OUTAGE.DURATION(mins)'. By looking at our dataset, we observed that 'OUTAGE.DURATION(mins)' was missing if and only if 'OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)' was also missing. Therefore, we assumed that the designers of the data collection process intentionally decided to not have data in 'OUTAGE.DURATION(mins)' column when 'OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)' was missing. Thus, the values in 'OUTAGE.DURATION(mins)' column were missing by design. Since the missing values in 'OUTAGE.DURATION(mins)' column were not a lot, only around 3.7%, dropping these values would not significantly introduce bias or affect the overall quality and representativeness of the data. Also, it would not significantly affect the results of the analysis. In this case, it might be acceptable to exclude the missing values in 'OUTAGE.DURATION(mins)' column from our analysis. Moreover, therew were 9 missing values in both the 'MONTH' and 'CLIMATE.CATEGORY' columns, located in the same rows. Most of the other data in these rows was also missing. In our case, we might assume that dropping these missing values in 'MONTH' and 'CLIMATE.CATEGORY' columns would not significantly change the data. The response variable column didn't contain nan, so we didn't need to perform any manipulation.

Since there were 443 missing values in 'CUSTOMERS.AFFECTED' column, we could not simply do list-wise deletion without looking at our data. We decided to do a permutation test to see if the missingness of CUSTOMER.AFFECTED depended on POSTAL.CODE. If the missingness of CUSTOMER.AFFECTED was dependent on POSTAL.CODE, we would do a conditional probabilistic imputation. If not, we would just do a probabilistic imputation. The null hypothesis was that the distribution of POSTAL.CODE when affected_customers is missing is the same as the distribution of POSTAL.CODE when affected_customers is not missing. The alternative hypothesis was that the missingness in the affected_customers column is dependent on POSTAL.CODE. After performing the permutation test, we found that the p-value was 0.0, so we rejected the null hypothesis and concluded that the missingness in the affected_customers column was dependent on POSTAL.CODE. Therefore, we would do a conditional probabilistic imputation. However, after exploring the dataset, we observed that for MT and SD, there contain only null values in 'CUSTOMERS.AFFECTED'. Since there was only 2 samples for SD and 3 samples for MT, meaning that dropping these values would not cause a huge difference in result or brought any bias, we decided to drop these rows.

In classification problems, evaluation metrics such as accuracy, precision, recall and F1-score are typically used to assess the performance of the model. Our goal here is to predict the cause of an outage. We assume that the cost of misclassification is the same for all classes. In other words, the cost of false negatives and false positives are the same. In this case, we can use accuracy or F1-score as our evaluation metric. However, the classes in our dataset are very imbalance, meaning that the number of samples in each class is not roughly equal. For example, we have 744 samples belong to severe weather outages, while there are only 38 samples belong to fuel supply emergency Outages. 

`print(outage["CAUSE.CATEGORY"].value_counts().to_markdown(index=True))`

|                               |   CAUSE.CATEGORY |
|:------------------------------|-----------------:|
| severe weather                |              763 |
| intentional attack            |              418 |
| system operability disruption |              127 |
| public appeal                 |               69 |
| equipment failure             |               60 |
| fuel supply emergency         |               51 |
| islanding                     |               46 |


In this case, using accuracy as an evaluation metric can be misleading, since a model that simply predicts the majority class would achieve a high accuracy. Therefore, using F-1 score as our evaluation metric is the most appropriate when dealing with class imbalance. In addition, since all causes are equally important, by setting the average parameter to "macro", we treat each class equally so that each class is equally important when predicting the result. 


------
# Baseline Model

For the baseline model, we used CLIMATE.CATEGORY, MONTH, and OUTAGE.DURATION(mins) as features to predit the cause of power outages. CLIMATE.CATEGORY is the climate episodes corresponding to the years. The categories—“Warm”, “Cold” or “Normal” episodes of the climate are based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI), and it's nominal data. MONTH is the month that a power outage happened, which is also a nominal data. OUTAGE.DURATION(mins) is the duration of a power outage, which is a quantitative data. 

We first used ColumnTransformer to manipulate these three columns by onehotencoding the CLIMATE.CATEGORY and MONTH column (since they are categorical variables), and standardizing the OUTAGE.DURATION(mins) column so that we could compare the duration time more effectively and in a meaningful way. We would drop other columns in the dataframe by setting the remainder to drop. Then, we put all these transformations into the pipeline and used DecisionTreeClassifier to train our model. 

Then we fit the pipeline to the training data and evaluated the F-1 score of both training and testing data. After splitting the dataset into training and testing data, the performance, or the model’s ability to generalize to unseen data, was around 0.97 for training data and only 0.25-0.35 for testing data. 

We believe that the F-1 score of the current model is not good, since we don't have any hyperparameter for the decision tree model, which will result in significant overfitting. Therefore, it doesn't generalize well for the unseen data. In addition, the F-1 score is low for testing data since we have only considered three features in our model, while in reality, the cause of a power outage is related to many different factors. In addition, the strength of correlations between different columns are not very strong after exploring dataset by drawing various graphs in project 3. 


------
# Final Model

For the final model 1, we added U.S._STATE and CUSTOMERS.AFFECTED as our new features. We believe that U.S._STATE is good for prediction because each state has different factors such as environments and policies that might affect the cause of power outage differently. CUSTOMERS.AFFECTED is likely related to the cause since certain causes might lead to more numbers of customers affected, and vice versa. Before we created column transformers, we first defined the class StdScalerByGroup that we created in lab 9 so that we could starndardize values based on groups. In the column transformers, in addition to the baseline model, we onehotencode the U.S._STATE, and used the parameter handle_unknown = 'ignore' to avoid error when a category is not found in the training dataset. We also used StdScalerByGroup to standardize CUSTOMERS.AFFECTED based on CLIMATE.CATEGORY. Lastly, we would keep other columns in the dataframe as they are by setting the remainder to passthrough. We used DecisionTreeClassifier as the model. There are several hyperparameters in the DecisionTreeClassifier that can be tuned to improve our model's performance. Here we decided to tune 'criterion', 'max_depth' and 'min_sample_split', which were commonly tuned hyperparameters.The hyperparameters that we chose from were: `hyperparameters = {'max_depth': range(1, 20), 'min_samples_split': range(2, 20),'criterion': ['gini', 'entropy']}` Then we used GridSearchCV to find the most optimal hyperparameters, which were: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 15}. The resulting F-1 score for final model 1 was about 0.25 for training data and 0.23 for testing data.

For the final model 2, we added TOTAL.CUSTOMERS as our new feature on top of final model 1. For the columntransformer, we standardized the TOTAL.CUSTOMERS, and everything was the same as final model 1. The optimal hyperparameters for this model was {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 10}. The resulting F-1 score for final model 2 was about 0.51 for training data and 0.34 for testing data.

For the final model 3, we added Day_of_week as our new feature on top of final model 1. For the columntransformer, we onehotencoded the TOTAL.CUSTOMERS, and everything was the same as final model 1. The optimal hyperparameters for this model was {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 7}. The resulting F-1 score for final model 3 was about 0.5 for training data and 0.35 for testing data.

We believe that the final model 3 works the best since the F-1 score improved the most in the last model and the day of week feature related more to the cause of the power outage. 

------
# Fairness Analysis

For our fairness analysis, we choose to split the dataset into two groups according to the YEAR column. One group consists of data before 2008, and the other group is data after 2008. The null hypothesis is that the final model 3's F-1 score is the same for the years before 2008 and years after 2008, and any differences are due to chance. The alternative hypothesis is that the final model 3's F-1 score is lower for years before 2008. The evaluation matric is F-1 score, and the test statistic is the difference in F-1 score (years before 2008 minus years after 2008). The significance level is 0.05, and the resulting p-value is 0.0. Therefore, we fail to reject the null hypothesis and conclude that the final model 3's F-1 score is the same for the years before 2008 and years after 2008, and any differences are due to chance.

