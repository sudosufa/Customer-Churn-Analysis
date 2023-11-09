#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis
# 

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random
random.seed(101)
data = pd.read_csv(r"C:/Users/Fatou Fall/Downloads/archive(2)/Customer-Churn-Records.csv")


# In[66]:


get_ipython().run_line_magic('autosave', '1')


# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


data.head()


# In[69]:


data.shape


# In[70]:


data.info()


# In[71]:


data.describe()


# In[72]:


data.isnull().sum()


# In[73]:


df = data.drop(columns={'RowNumber', 'CustomerId', 'Surname'}, axis=1)


# In[74]:


df.head()


# In[75]:


df.duplicated().sum()


# In[76]:


df['Age'].min()


# # Number of customers by Geography and Age Group:
# 

# In[77]:


import pandas as pd

# Create age groups (e.g., 20-30, 31-40, etc.)
age_bins = [0, 20, 30, 40, 50, 60, 70]
age_labels = ['0-20', '20-30', '31-40', '41-50', '51-60', '61+']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Aggregate number of customers by geography and age group
No_of_customers_geo_age = df.groupby(['Geography', 'Age Group']).size().reset_index(name='Number of Customers')

# Sort the aggregated data by the number of customers in ascending order
No_of_customers_geo_age_sorted = No_of_customers_geo_age.sort_values(by='Number of Customers', ascending=True)

# Display the sorted aggregated data
print(No_of_customers_geo_age_sorted)


# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create the grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=No_of_customers_geo_age, x='Age Group', y='Number of Customers', hue='Geography')
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.title('Number of Customers by Age Group (Grouped by Geography)')
plt.tight_layout()
plt.show()


# # Customer Count by Card Type and IsActiveMember:
# 

# In[79]:


df.info()


# In[83]:


# Aggregate customer count by card type and active membership
customer_count_card_member = df.groupby(['Card Type', 'IsActiveMember']).size().reset_index(name='CustomerCount')
# Display the aggregated data
#print(customer_count_card_member)


# In[85]:


import seaborn as sns
import matplotlib.pyplot as plt

# Your previous code for customer_count_card_member...

# Display the aggregated data
print(customer_count_card_member)

# Plot the data using seaborn
sns.barplot(data=customer_count_card_member, x='Card Type', y='CustomerCount', hue='IsActiveMember')
plt.show()


# # Distribution of Age

# In[86]:


# Plot a histogram of the 'Age' feature
plt.figure(figsize =(10,6))
plt.hist(df['Age'], bins=30)
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.title('Distribution of Age')
plt.show()


# Customers between age 26-45 are the most
# Cause: This age group represents the working population and individuals who are likely to have
# banking needs such as loans, mortgages, and investments.
# 

# # Distribution of Gender

# In[87]:


# Plot a bar chart of the 'Gender' feature
plt.figure(figsize =(10,6))
gender_counts = df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.title('Distribution of Gender')
plt.show()


# Not a very big differnce between gender distribution of customers.
# 

# # Distribution of Card Type

# In[88]:


# Plot a pie chart of the 'Card Type' feature
plt.figure(figsize =(12,8))
card_type_counts = df['Card Type'].value_counts()
plt.pie(card_type_counts, labels=card_type_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Card Type')
plt.show()


# Almost all the card types have equal share of customers.
# Cause: The distribution of card types suggests that customers have diverse preferences, and the
# bank has effectively marketed and provided options for different card types to cater to customer
# needs.

# # Correlation Matrix

# In[90]:


# Select only the numerical features for correlation matrix
numerical_features = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts']]

# Compute the correlation matrix
corr_matrix = numerical_features.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Not a very good correlations between features
# Finding: The correlation analysis reveals that there is no strong linear relationship between the
# features examined.
# Cause: The absence of strong correlations suggests that customer churn may be influenced by
# a combination of multiple factors rather than a single dominant factor.

# # Churn Rate by Geography

# In[91]:


# Plot the churn rate by 'Geography'
plt.figure(figsize =(10,6))
churn_rate_geo = df.groupby('Geography')['Exited'].mean()
plt.bar(churn_rate_geo.index, churn_rate_geo.values)
plt.xlabel('Geography')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Geography')
plt.show()


# Germany has highest chuned customers
# Cause: It could be due to various reasons such as customer dissatisfaction, competitive offers from
# other banks, or specific market conditions in Germany.

# # Churn Rate by Age and Gender

# In[92]:


import plotly.express as px
# Calculate churn rate by age and gender
churn_rate_age_gender = df.groupby(['Age', 'Gender'])['Exited'].mean().reset_index()
# Create an interactive scatter plot
fig = px.scatter(churn_rate_age_gender, x='Age', y='Exited', color='Gender',
 title='Churn Rate by Age & Gender')
fig.update_layout(xaxis_title='Age', yaxis_title='Churn Rate')
fig.show()


# From the scatter plot, it can be seen that
# 1. Customers of age between 45 - 70 have higher tendency to leave the bank.
# 2. Female has higher churn rate than male.
# Cause:
# A. Older customers may be more likely to consider switching banks due to life changes,
# retirement, or seeking better financial products and services.
# B. The higher churn rate among female customers could be influenced by various factors such as
# customer satisfaction, service quality, targeted marketing strategies, or specific life events.
# 

# # Churn Rate by Tenure & HasCrCard:
# 

# In[94]:


# Aggregate churn rate by tenure and credit card status
churn_rate_tenure_card = data.groupby(['Tenure', 'HasCrCard'])['Exited'].mean().reset_index(name='ChurnRate')

# Display the aggregated data
#print(churn_rate_tenure_card)


# In[96]:


# Create an interactive scatter plot
fig = px.scatter(churn_rate_tenure_card, x='Tenure', y='ChurnRate', color='HasCrCard',
                 title='Churn Rate by Tenure & Has Credit Card')
fig.update_layout(xaxis_title='Tenure', yaxis_title='Churn Rate')
fig.show()


# # Churn Rate by Geography and Tenure

# In[98]:


# Aggregate churn rate by tenure and credit card status
churn_rate_tenure_card = data.groupby(['Tenure', 'HasCrCard'])['Exited'].mean().reset_index(name='ChurnRate')

# Display the aggregated data
#print(churn_rate_tenure_card)


# In[100]:


# Calculate churn rate by geography and tenure
churn_rate_geo_tenure = df.groupby(['Geography', 'Tenure'])['Exited'].mean().reset_index(name='ChurnRate')

# Create an interactive bar chart
fig = px.bar(churn_rate_geo_tenure, x='Geography', y='ChurnRate', color='Tenure',
             title='Churn Rate by Geography & Tenure')
fig.update_layout(xaxis_title='Geography', yaxis_title='Churn Rate')
fig.show()


# As you can see, old customers are the one who are leaving the bank most.
# Cause: Older customers may be more aware of their banking needs and have higher expectations
# regarding customer service, financial advice, and personalized offerings. If these expectations are
# not met, they may choose to switch banks.

# # Churn Rate by Credit Score and NumOfProducts

# In[101]:


# Calculate churn rate by credit score and number of products
churn_rate_credit_products = df.groupby(['CreditScore', 'NumOfProducts'])['Exited'].mean().reset_index(name='ChurnRate')

# Create an interactive heatmap
fig = px.density_heatmap(churn_rate_credit_products, x='CreditScore', y='NumOfProducts',
                         z='ChurnRate', title='Churn Rate by Credit Score & No. Of Products')
fig.update_layout(xaxis_title='Credit Score', yaxis_title='Number Of Products')
fig.show()


# From the heatmap, it is clear that customers with 3 Num. of Products and with a credit score
# between 550 - 700 are the most amongst the churned customers.
# Cause: Customers with multiple products and a moderate credit score may be more sensitive to
# changes in service quality, pricing, or other competitive factors. They may also be targeted by
# competitors offering attractive incentives to switch banks.

# In[102]:


df.sample(5)


# In[103]:


df = df.drop('Age Group', axis =1)
df


# # Building ML Predictive Model

# In[105]:


# Step 1: Data Preprocessing
# Handle Missing Values if any
df.fillna(method='ffill', inplace=True)
# Convert Categorical Columns to 'category' data type
df['Geography'] = df['Geography'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Card Type'] = df['Card Type'].astype('category')
# Encode Categorical Variables
encoded_data = pd.get_dummies(df, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)
# Scale Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)
# Step 2: Feature Engineering
# Feature Selection
selected_features = encoded_data.drop(['Exited'], axis=1)
target_variable = encoded_data['Exited']


# In[107]:


# Step 3: Model Selection and Training
# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(selected_features, target_variable, test_size=0.2, random_state=42)

# Train and Evaluate Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# In[108]:


# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
# Calculate Precision
precision = precision_score(y_test, rf_predictions)
# Calculate Recall
recall = recall_score(y_test, rf_predictions)
# Calculate F1-score
f1 = f1_score(y_test, rf_predictions)
# Print the results
print("Accuracy score:",rf_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:


# Step 4: Hyperparameter Tuning
# Optimize Model Performance
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search Cross Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_

# Step 5: Model Evaluation
# Compare Model Performance
grid_search_predictions = grid_search.predict(X_test)
grid_search_accuracy = accuracy_score(y_test, grid_search_predictions)
grid_search_F1_Score = f1_score(y_test, grid_search_predictions)
print("Grid Search Accuracy:", grid_search_accuracy)
print("Grid Search F1-Score:", grid_search_F1_Score)


# Despite our attempts at Hyperparameter Tuning, it appears that we were unable to enhance the
# performance of the model.
# However, it's worth noting that the model's accuracy of 99.9% and F1-Score of 0.9974 indicates a
# highly efficient model.

# In[ ]:


best_params


# In[ ]:


# Step 6: Deployment
# Deploy the Final Model
final_model = grid_search.best_estimator_
final_model.fit(selected_features, target_variable)


# To deploy the model for a new data,following steps can be followed1. Read new data: new_data = pd.read_csv('new_data.csv')
# 2. Preprocess the new data
# 3. Apply feature engineering
# 4. Scale the features
# 5. Use the final_model.predict() function to make predictions

# # Recommendations for the Bank

# 1. Improve Customer Satisfaction: Conduct regular customer satisfaction surveys to identify pain
# points and areas for improvement. Addressing customer concerns promptly and effectively can help
# reduce churn rates.
# 2. Enhance Customer Retention Programs: Develop loyalty programs, personalized offers, and
# rewards to incentivize customers to stay with the bank. Building strong relationships and providing
# value-added services can increase customer loyalty.
# 3. Focus on Retaining Female Customers: Analyze the reasons behind the higher churn rate
# among female customers. Tailor marketing and customer service strategies to meet their specific
# needs and preferences.
# 4. Strengthen Communication Channels: Ensure effective communication channels are in place to
# keep customers informed about new products, services, and updates. Regularly engage with
# customers through personalized interactions and provide timely support.
# 5. Offer Targeted Financial Solutions: Analyze the needs and preferences of customers within the
# age group of 45-70. Provide tailored financial solutions, such as retirement planning, investment
# options, and specialized services, to meet their unique requirements.
# 6. Provide Value-added Products and Services: Continuously assess the market to identify
# emerging trends and offer innovative products and services that differentiate the bank from
# competitors. Regularly review and update existing offerings to remain competitive.
# 7. Foster Trust and Transparency: Build trust and transparency through clear communication, fair
# pricing, and reliable services. Ensure customers feel valued and have confidence in the bank's
# integrity and commitment to their financial well-being.
# 8. Focus on Customer Education: Offer financial literacy programs and educational resources to
# empower customers to make informed decisions. Educated customers are more likely to remain
# loyal and satisfied with the bank's services.
# By implementing these recommendations, the bank can strengthen customer relationships,
# increase customer satisfaction, and ultimately reduce churn rates. Regular monitoring and analysis
# of customer behavior and feedback will help refine strategies and further improve customer
# retention efforts.

# In[ ]:




