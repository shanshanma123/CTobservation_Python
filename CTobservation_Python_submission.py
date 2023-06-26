#!/usr/bin/env python
# coding: utf-8

# In[2]:


#CTobservation study 
#Three data sets: academic_performance, jigsaw_performance, and CT_Performance
#three groups, group A no hints (control group), group B encouraging words, group C hints locating pieces 

# import necessary libraries
import pandas as pd 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import statsmodels.api as sm


# In[8]:


#Get the working directory 
import os 
os.getcwd()


# In[4]:


#Read the file Academic_Performance.csv
df1_Academic = pd.read_csv('/Users/shanshanma/Desktop/CTObservation_Python/Academic_Performance.csv')
df1_Academic.info()


# In[5]:


#Remove cases with missing values 
df1_acaclean = df1_Academic.dropna(axis=0, how='any')
df1_acaclean.info()


# In[11]:


#Descriptive statistics 
df1_acaclean.describe()


# In[8]:


#Read the file Jigsaw_Performance.csv
df2_Jigsaw = pd.read_csv('/Users/shanshanma/Desktop/CTObservation_Python/Jigsaw_Performance.csv')
df2_Jigsaw.info()
df2_Jigsaw.head(5)


# In[9]:


#Remove cases with missing values 
df2_jsclean = df2_Jigsaw.dropna(axis=0, how='any')


# In[14]:


df2_jsclean.info()


# In[7]:


#Check the number of participants in each group 
df2_jsclean_tab = pd.crosstab(index=df2_jsclean["Group"],
                               columns="count")
df2_jsclean_tab


# In[16]:


#Descriptive statistics
df2_jsclean.describe()


# In[17]:


#Filter numeric columns only
numeric_df2_js = df2_jsclean.select_dtypes(include='number')
numeric_df2_js.info()


# In[18]:


#Check the correlation coefficient and significance among variables in df2 
for col1 in numeric_df2_js.columns:
    for col2 in numeric_df2_js.columns:
        if col1 != col2:
            correlation_coefficient, p_value = stats.pearsonr(numeric_df2_js[col1], numeric_df2_js[col2])
            print(f"Correlation coefficient between {col1} and {col2}: {correlation_coefficient}")
            print(f"P-value: {p_value}\n")


# In[ ]:


#Recoding Group value from numeric to string
df2_jsclean['Group'].replace({1: 'A', 2: 'B', 3: 'C'}, inplace= True)
df2_jsclean.info()


# In[20]:


#Create histograms by Group A,B,C
df2_jsclean['Duration'].hist(by=df2_jsclean['Group'], edgecolor='black',figsize=(8,6))


# In[21]:


#A summary of jigsaw puzzle solving time by group
rp.summary_cont(df2_jsclean['Duration'])
rp.summary_cont(df2_jsclean['Duration'].groupby(df2_jsclean['Group']))


# In[22]:


#one-way ANOVA
stats.f_oneway(df2_jsclean['Duration'][df2_jsclean['Group'] == 'A'],
               df2_jsclean['Duration'][df2_jsclean['Group'] == 'B'],
               df2_jsclean['Duration'][df2_jsclean['Group'] == 'C'])


# In[23]:


#correlations between jigsaw puzzle solving variables 
correlations_js = df2_jsclean[["Duration", "StrategicOptions","ObservationMode","Complete","Tipscheck"]].corr()
correlations_js


# In[24]:


#regression 1: to what extent the rest of variables can predict time needed for jigsaw puzzle solving 
x = df2_jsclean[['ObservationMode', 'StrategicOptions','Complete','Tipscheck']]
y = df2_jsclean['Duration']
x = sm.add_constant(x)
reg1 = sm.OLS(y, x).fit()
print(reg1.summary())


# In[27]:


#A scatter plot for the correlation between variables Tipscheck and Duration
plt.scatter(df2_jsclean['Tipscheck'], df2_jsclean['Duration'], color='green')
plt.title('Tipscheck Vs Time', fontsize=12)
plt.xlabel('Tipscheck', fontsize=12)
plt.ylabel('Time', fontsize=12)
plt.grid(True)
plt.show()


# In[28]:


#A scatter plot for the correlation between variables StrategicOptions and Duration
plt.scatter(df2_jsclean['StrategicOptions'], df2_jsclean['Duration'], color='green')
plt.title('StrategicOptions Vs Time', fontsize=12)
plt.xlabel('StrategicOptions', fontsize=12)
plt.ylabel('Time', fontsize=12)
plt.grid(True)
plt.show()


# In[29]:


#A scatter plot for the correlation between variables ObservationMode and Duration
plt.scatter(df2_jsclean['ObservationMode'], df2_jsclean['Duration'], color='green')
plt.title('ObservationMode Vs Time', fontsize=12)
plt.xlabel('Observation Mode', fontsize=12)
plt.ylabel('Time', fontsize=12)
plt.grid(True)
plt.show()


# In[85]:


#Read file CT_Performance.csv
df3_CT = pd.read_csv('/Users/shanshanma/Desktop/CTObservation_Python/CT_Performance.csv')
df3_CT.info()
df3_CT.head(5)


# In[86]:


#Remove cases with missing values
df3_CTclean = df3_CT.dropna(axis=0, how='any')
df3_CTclean.info()


# In[87]:


#Descriptive statistics
df3_CTclean.describe()


# In[88]:


#Select numeric data only
numeric_df3CT = df3_CTclean.select_dtypes(include='number')

#Perform correlation analysis - correlation coefficient only
correlation_df3CT = numeric_df3CT.corr()
print(correlation_df3CT)


# In[89]:


#Check correlation coefficient and significance among variables in df3
for col1 in numeric_df3CT.columns:
    for col2 in numeric_df3CT.columns:
        if col1 != col2:
            correlation_coefficient, p_value = stats.pearsonr(numeric_df3CT[col1], numeric_df3CT[col2])
            print(f"Correlation coefficient between {col1} and {col2}: {correlation_coefficient}")
            print(f"P-value: {p_value}\n")


# In[10]:


#Merge data sets df1 and df2
merged_acajs = pd.merge(df1_acaclean, 
                        df2_jsclean,
                       how = "inner",
                       on="Code")
merged_acajs


# In[78]:


#Select numeric data only
numeric_merged_acajs = merged_acajs.select_dtypes(include='number')
numeric_merged_acajs.head(5)


# In[79]:


#Perform correlation analysis within the merged data set numeric_merged_acajs
correlation_merged_acajs = numeric_merged_acajs.corr()
print(correlation_merged_acajs)


# In[94]:


#Merge data sets df1 and df3
merged_acact = pd.merge(df1_acaclean, 
                        df3_CTclean,
                       how = "inner",
                       on="Code")
merged_acact


# In[95]:


#Select numeric data only - merged_acact
numeric_merged_acact = merged_acact.select_dtypes(include='number')
numeric_merged_acact.head(5)


# In[105]:


#Perform correlation analysis within the merged data set merged_acact
correlation_merged_acact = numeric_merged_acact.corr()
print(correlation_merged_acact)


# In[117]:


#Regression 2: to what extent the academic performance can predict students'induction ability
x = numeric_merged_acact[['CN9','MTH9', 'Eng9','CN10','MTH10','Eng10']]
y = numeric_merged_acact['Part1_Sum']
x = sm.add_constant(x)
reg2 = sm.OLS(y, x).fit()
print(reg2.summary())


# In[118]:


#regression 3: to what extent the academic performance can predict students'ability in credibility of Sources&Observation
x = numeric_merged_acact[['CN9','MTH9', 'Eng9','CN10','MTH10','Eng10']]
y = numeric_merged_acact['Part2_Sum']
x = sm.add_constant(x)
reg3 = sm.OLS(y, x).fit()
print(reg3.summary())


# In[119]:


#regression 4: to what extent the academic performance can predict students'deduction ability
x = numeric_merged_acact[['CN9','MTH9', 'Eng9','CN10','MTH10','Eng10']]
y = numeric_merged_acact['Part3_Sum']
x = sm.add_constant(x)
reg4 = sm.OLS(y, x).fit()
print(reg4.summary())


# In[120]:


#regression 5: to what extent the academic performance can predict students'ability in assumption identification. 
x = numeric_merged_acact[['CN9','MTH9', 'Eng9','CN10','MTH10','Eng10']]
y = numeric_merged_acact['Part4_Sum']
x = sm.add_constant(x)
reg5 = sm.OLS(y, x).fit()
print(reg5.summary())


# In[ ]:


#regression 6: to what extent the academic performance can predict students'CT performance
x = numeric_merged_acact[['CN9','MTH9', 'Eng9','CN10','MTH10','Eng10']]
y = numeric_merged_acact['All_Sum']
x = sm.add_constant(x)
reg6 = sm.OLS(y, x).fit()
print(reg6.summary())


# In[101]:


#merge data sets df2 and df3
merged_jsct = pd.merge(df2_jsclean, 
                        df3_CTclean,
                       how = "inner",
                       on="Code")
merged_jsct


# In[102]:


#Select numeric data only - merged_jsct
numeric_merged_jsct = merged_jsct.select_dtypes(include='number')
numeric_merged_jsct.head(5)


# In[104]:


#Perform correlation analysis within the merged data set merged_jsct
correlation_merged_jsct = numeric_merged_jsct.corr()
print(correlation_merged_jsct)


# In[110]:


#Regression 7: to what extent the jigsaw performance can predict students' CT performance
x = numeric_merged_jsct[['Duration','ObservationMode', 'StrategicOptions','Complete','Tipscheck']]
y = numeric_merged_jsct['All_Sum']
x = sm.add_constant(x)
reg7 = sm.OLS(y, x).fit()
print(reg7.summary())


# In[ ]:




