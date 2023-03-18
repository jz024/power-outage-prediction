# power outage prediction
by Tongxun Hu and Jessica Zhang

# Framing the Problem

For this power outage dataset, we try to predict the cause of a major power outage. We will use the already cleaned data from project 3. We are trying to make prediction for the major power outages after 2006, so that we assume that we can access the entire dataframe of power outage (range from 2000 to 2006) to train our model. The response variable will be the CAUSE.CATEGORY column in the dataset; we choose it because we think that it is closely related to the power outage, so it will predict the power outage in an effective way. This prediction will be a classification and we will use multi-class classification since there are multiple causes of power outages. We choose to use accuracy to evaluate our model because it measures the overall performance of the prediction based on the entire set of data. Each cause weighs the same so accuracy can be used to make sure that they contribute equally when calculating the accuracy value. 


------
# Baseline Model

For the baseline model, we used CLIMATE.CATEGORY, MONTH, and OUTAGE.DURATION(mins) as features to predit the cause of power outages. CLIMATE.CATEGORY is the climate episodes corresponding to the years. The categories—“Warm”, “Cold” or “Normal” episodes of the climate are based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI), and it's nominal data. MONTH is the month that a power outage happened, which is also a nominal data. OUTAGE.DURATION(mins) is the duration of a power outage, which is a quantitative data. 

We first used ColumnTransformer to manipulate these three columns by onehotencode the CLIMATE.CATEGORY and MONTH column (since they are categorical variables), and standardize the OUTAGE.DURATION(mins) column so that we can compare the duration time more effectively and in a meaningful way. We will drop other columns in the dataframe by setting the remainder to drop. Then, we put all these transformations into the pipeline and used LogisticRegression to train our model. Lastly, we fit the pipeline to the training data and evaluated the accuracy of both training and testing data. After splitting the dataset into training and testing data, the performance, or the model’s ability to generalize to unseen data, is around 0.6 for the testing data, and 0.63 for the training data. 

We believe that the accuracy of the current model is reasonable, but not very good, since we had only considered three features in our model, while in reality, the cause of a power outage is related to many different factors. 


------
# Final Model


------
# Fairness Analysis



