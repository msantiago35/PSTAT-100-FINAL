#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from pathlib import Path
import os
import sklearn.linear_model as lm
from sklearn.preprocessing import add_dummy_feature
from sklearn.metrics import r2_score
alt.data_transformers.disable_max_rows()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

GDIMoriginal = pd.read_csv('GDIM_2021_09.csv')
GDIMoriginal
EqualityData = pd.read_csv('csvData.csv')
EqualityData.columns
EqualityData=EqualityData.rename(columns={'gendEqal2021':'GEI %', 'econ':'Gender Economic Participation and Opportunity Equality %', 'education':'Gender Education Attainment Equality %', 'health':'Gender Health and Survival Equality %', 'polit':'Gender Political Empowerment'})
region = pd.read_csv('all.csv')
region
#http://info.worldbank.org/governance/wgi/Home/Reports
politicalstability=pd.read_csv('Political_StabilityNoViolence-Table_1.csv')
politicalstability=politicalstability.rename(columns={'Country/Territory':'country'})
politicalstability=pd.melt(politicalstability, id_vars=['country'], value_vars=['1996', '1998', '2000', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
politicalstability=politicalstability.rename(columns={'variable':'year', 'value':'Political Instability Estimate'})
politicalstability
#https://www.worlddata.info/iq-by-country.php
IQ=pd.read_csv('IQ_data_-_Sheet1.csv')
IQ
#https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv
#https://worldpopulationreview.com/country-rankings/gender-equality-by-country
region.head()
region=region.rename(columns={'name':'country'})
region['country'] = region['country'].replace(['United Kingdom of Great Britain and Northern Ireland'],'United Kingdom')
region['country'] = region['country'].replace(['United States of America'],'United States')
region['country'] = region['country'].replace(['Bolivia (Plurinational State of)'],'Bolivia')
region['country'] = region['country'].replace(['Viet Nam'],'Vietnam')
region=region[['country','sub-region']]
EqualityData.head()
os.getcwd()
#joining the two data sets
GDIMjoined=GDIMoriginal.join(EqualityData.set_index('country'), on='country')
GDIMjoined=GDIMjoined.join(region.set_index('country'), on='country')
GDIMoriginal['year']=GDIMoriginal['year'].astype(int)
politicalstability['year']=politicalstability['year'].astype(int)
GDIMjoined = GDIMjoined.merge(politicalstability, on=['country', 'year'])
GDIMjoined = GDIMjoined.merge(IQ, on=['country'])
GDIMjoined

#total in politcal instability estimate that are null
GDIMjoined['Political Instability Estimate'].isnull().sum()

#CAT = measure of absolute mobility

#ICSED = International Standard Classification of Education (ISCED): less than primary
#(ISCED 0), primary (ISCED 1), lower secondary (ISCED 2), upper secondary or postsecondary
#non-tertiary (ISCED 3–4), and tertiary (ISCED 5–8). The categories refer to the highest educational
#level completed by the respondent.7

# Dropping columns that won't be used
GDIMdropped=GDIMjoined.drop(['code','year','region','COR','YOS','MIX','MLD_psu','incgroup2','incgroup3','fragile','survey','status','cohort','obs','P1', 'P2', 'P3', 'P4', 'P5', 'C1', 'C2', 'C3', 'C4', 'C5', 'BETA', 'MU050' ,'BHQ4','Q4Q4' , 'BHQ1', 'BHQ2', 'BHQ3', 'Q4BH', 'Q4child','CAT_ISCED5678'], axis=1
                           ).rename(columns ={'region_noHICgroup':'region'})

GDIM=GDIMdropped.dropna()

GDIMdropped.corr()



#Grouped by Region of world
GDIM.groupby('sub-region').mean().reset_index()
GDIM.loc[GDIM['child']=='daughter'].groupby('sub-region').mean().reset_index()
GDIM.loc[GDIM['child']=='son'].groupby('sub-region').mean().reset_index()

#seperation of sexes of children
daughterGDIM=GDIM.loc[GDIM['child']=='daughter']
sonGDIM=GDIM.loc[GDIM['child']=='son']
allGDIM=GDIM.loc[GDIM['child']=='all']
#________________________________________________
#seperation of sexes for children and parents
daughterGDIMdad=daughterGDIM.loc[GDIM['parent']=='dad']
daughterGDIMmom=daughterGDIM.loc[GDIM['parent']=='mom']
daughterGDIMavg=daughterGDIM.loc[GDIM['parent']=='avg']

sonGDIMdad=sonGDIM.loc[GDIM['parent']=='dad']
sonGDIMmom=sonGDIM.loc[GDIM['parent']=='mom']
sonGDIMavg=sonGDIM.loc[GDIM['parent']=='avg']

allGDIMdad=allGDIM.loc[GDIM['parent']=='dad']
allGDIMmom=allGDIM.loc[GDIM['parent']=='mom']
allGDIMavg=allGDIM.loc[GDIM['parent']=='avg']


# In[2]:


region.head()


# In[3]:


GDIM.head()


# In[4]:


GDIM['Gender Education Attainment Equality %'].max


# In[5]:


GDIM.loc[:, 'country':].isna().sum().reset_index().rename(columns={'index': 'variable', 0: 'missing values'})


# In[6]:


pop_plot = alt.Chart(daughterGDIM).mark_bar().encode(
    x = alt.X('incgroup4:N',sort='-y'), y = alt.Y('MEANc', scale = alt.Scale(zero = False))
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layereddaughter=alt.layer(pop_plot, data=daughterGDIM).facet(column= 'parent')



layereddaughter


# In[7]:


#there is an obvious increase in 


# In[8]:



pop_plot = alt.Chart(sonGDIM).mark_bar().encode(
    x = alt.X('incgroup4:N', sort='-y'), y = alt.Y('MEANc', scale = alt.Scale(zero = False))
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layeredson=alt.layer(pop_plot, data=sonGDIM).facet(column= 'parent')



layeredson


# In[9]:


pop_plot = alt.Chart(allGDIM).mark_bar().encode(
    x = alt.X('incgroup4:N', sort='-y'), 
    y = alt.Y('MEANc', scale = alt.Scale(zero = False))
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layeredall=alt.layer(pop_plot, data=allGDIM).facet(column= 'parent')



layeredall


# In[15]:


pop_plot = alt.Chart(allGDIM).mark_bar().encode(
    x = alt.X('incgroup4:N', sort='-y'), 
    y = alt.Y('MEANc', scale = alt.Scale(zero = False)),
    color = alt.Color('region')
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layeredall=alt.layer(pop_plot, data=allGDIM).facet(column= 'parent')



layeredall

#use this in analysis 


# In[11]:


GDIMsubregion=GDIM.groupby('sub-region').mean().reset_index()


# In[12]:


GDIMsubregion=GDIM.groupby('sub-region').mean().reset_index()
pop_plot = alt.Chart(GDIMsubregion).mark_bar().encode(
    x = alt.X('sub-region:N', sort='-y'), y = alt.Y('MEANc', scale = alt.Scale(zero = False))
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layeredall=alt.layer(pop_plot, data=GDIMsubregion).facet(column= 'parent')



layeredall

#use this in analysis 


# In[16]:


GDIMregion=GDIM.groupby('region').mean().reset_index()

pop_plot = alt.Chart(GDIMregion).mark_bar().encode(
    x = alt.X('region:N', sort='-y'), y = alt.Y('MEANc', scale = alt.Scale(zero = False))
).properties(width = 150,height = 200)
#.facet(column = 'incgroup4')

layeredall=alt.layer(pop_plot, data=GDIMregion).facet(column= 'parent')



layeredall

#use this in analysis 


# In[35]:


# store quantitative variables separately

x_mx = GDIM[['MEANc','MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
corr_mx = x_mx.corr()


# In[36]:


# correlation between education and other variables
corr_mx.loc[:, 'MEANc'].sort_values()


# Negatively correlated with Daily maximum\ntemperature Celsiu meaning it tends to vary in opposition with these variables;
# 
# Strongly positively correlated with MEANp, and IQ meaning it tends to vary together with these variables most.

# In[37]:


# store quantitative variables separately
# correlation matrix
allGDIMavg.head()
allGDIMavg.corr()


# store correlation matrix
corr_mx = x_mx.corr()

# melt to long form
corr_mx_long = corr_mx.reset_index().rename(
    columns = {'index': 'row'}
).melt(
    id_vars = 'row',
    var_name = 'col',
    value_name = 'Correlation'
)

# visualize
heatmap = alt.Chart(corr_mx_long).mark_rect().encode(
    x = alt.X('col', title = '', sort = {'field': 'Correlation', 'order': 'ascending'}),
    y = alt.Y('row', title = '', sort = {'field': 'Correlation', 'order': 'ascending'}),
    color = alt.Color('Correlation', 
                      scale = alt.Scale(scheme = 'blueorange',
                                        domain = (-1, 1), 
                                        type = 'sqrt'),
                     legend = alt.Legend(tickCount = 5))
).properties(width = 200, height = 200)


# In[41]:


heatmap


# In[42]:


#PCA 
# center and scale ('normalize')
x_ctr = (x_mx - x_mx.mean())/x_mx.std()
x_ctr


# In[43]:


# compute principal components
pca = PCA(n_components = x_ctr.shape[1]) 
pca.fit(x_ctr)


# In[44]:


# variance ratios
pca.explained_variance_ratio_


# In[45]:


# store proportion of variance explained as a dataframe
pca_var_explained = pd.DataFrame({'Proportion of variance explained': pca.explained_variance_ratio_})

# add component number as a new column
pca_var_explained['Component'] = np.arange(1, 10)

# print
pca_var_explained.head()


# In[46]:


# add component number as a new column
pca_var_explained['Cumulative variance explained'] = pca_var_explained.iloc[:,0].cumsum(axis = 0)

# print
pca_var_explained.head(3)


# In[47]:


# encode component axis only as base layer
base = alt.Chart(pca_var_explained).encode(
    x = 'Component')

# make a base layer for the proportion of variance explained
prop_var_base = base.encode(
    y = alt.Y('Proportion of variance explained',
              axis = alt.Axis(titleColor = '#57A44C'))
)

# make a base layer for the cumulative variance explained
cum_var_base = base.encode(
    y = alt.Y('Cumulative variance explained', axis = alt.Axis(titleColor = '#5276A7'))
)

# add points and lines to each base layer
prop_var = prop_var_base.mark_line(stroke = '#57A44C') + prop_var_base.mark_point(color = '#57A44C')
cum_var = cum_var_base.mark_line() + cum_var_base.mark_point()

# layer the layers
var_explained_plot = alt.layer(prop_var, cum_var).resolve_scale(y = 'independent')

# display
var_explained_plot


# In[48]:


#principal components explain more than 60% of total variation
main = pca_var_explained[pca_var_explained['Proportion of variance explained'] > 0.07].count()

main_pca = main[0] 

#print
main_pca 


# In[51]:


#How much total variation is captured by these 3 PC
main = pca_var_explained[pca_var_explained['Proportion of variance explained'] > 0.07].sum()
main_variation = main[0]

#print
main_variation


# More than 84% of the variation is captured by the first 3 components

# In[52]:


corr_mx = x_mx.corr()

# store the loadings as a data frame with appropriate names
loading_df = pd.DataFrame(pca.components_).transpose().rename(
    columns = {0: 'PC1', 1: 'PC2', 2: 'PC3' } # add entries for each selected component
).loc[:, ['PC1', 'PC2', 'PC3']] # slice just components of interest

# add a column with the variable names
loading_df['Variable'] = x_mx.columns.values

# print
loading_df.head()


# In[84]:


# melt from wide to long
loading_plot_df = loading_df.melt(
    id_vars = 'Variable',
    var_name = 'Principal Component',
    value_name = 'Loading'
)

# add a column of zeros to encode for x = 0 line to plot
loading_plot_df['zero'] = np.repeat(0, len(loading_plot_df))

# create base layer
base = alt.Chart(loading_plot_df)

# create lines + points for loadings
loadings = base.mark_line(point = True).encode(
    y = alt.X('Variable', title = ''),
    x = 'Loading',
    color = 'Principal Component'
)

# create line at zero
rule = base.mark_rule().encode(x = alt.X('zero', title = 'Loading'), size = alt.value(0.05))

# layer
loading_plot = (loadings + rule).properties(width = 120)

# show
loading_plot


# In[85]:


loading_plot.facet('Principal Component')


# - PC1 positively influenced by 'Daily maximum\ntemperature Celsius' and negatively influenced by everything else.
# - PC2 strongly positively influenced by 'Average Income (USD)',Education expenditure\nper inhabitant (USD)',and 'Gender Political Empowerment' and negatively influenced by 'almost everything else'.
# - PC3 strongly positively influenced by 'MEANp','MEANc','IQ' and negatively influenced by 'Gender Health and Survival Equality %', and 'GEI%'.  
# 
# Let's call,
# 
# PC1 'Temperature'.
# 
# PC2: 'Income and educational expenditure'. 
# 
# PC3: 'Intelligence'

# In[86]:


# three largest variances
x_mx.var().sort_values(ascending = False).head(3)


# In[88]:


# project data onto first four components; store as data frame
projected_data = pd.DataFrame(pca.fit_transform(x_ctr)).iloc[:, 0:4].rename(columns = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4'})

# add state and county
projected_data[['country', 'region']] = GDIM[['country', 'region']]

# print
projected_data.head(4)


# In[96]:


# base chart
base = alt.Chart(projected_data)

# data scatter
scatter = base.mark_point(opacity = 0.2).encode(
    x = alt.X('PC2:Q', title = 'Income and educational expenditure PC'),
    y = alt.Y('PC3:Q', title = 'Intelligence PC'),
)

# show
scatter


# In[97]:


# find cutoff value
sum = projected_data.PC2 + projected_data.PC3
pc_2plus4 = sorted(sum, reverse = True)

#print
pc_2plus4[0:15]


# In[108]:


# store outlying rows using cutoff
outliers = projected_data[(projected_data.PC2 + projected_data.PC3) > 5]

# plot outliers in red
pts = alt.Chart(outliers).mark_circle(
    color = 'red',
    opacity = 0.6
).encode(
    x = 'PC2',
    y = 'PC3'
)

# layer
scatter + pts


# In[99]:


# base chart
base = alt.Chart(projected_data)

# data scatter
scatter = base.mark_point(opacity = 0.2).encode(
    x = alt.X('PC1:Q', title = 'Temperature PC'),
    y = alt.Y('PC3:Q', title = 'Intelligence PC'),
)

# show
scatter


# In[100]:


# find cutoff value
sum = projected_data.PC1 + projected_data.PC3
pc_2plus4 = sorted(sum, reverse = True)

#print
pc_2plus4[0:15]


# In[101]:


# store outlying rows using cutoff
outliers = projected_data[(projected_data.PC1 + projected_data.PC3) > 9]

# plot outliers in red
pts = alt.Chart(outliers).mark_circle(
    color = 'red',
    opacity = 0.6
).encode(
    x = 'PC1',
    y = 'PC3'
)

# layer
scatter + pts


# In[109]:


# base chart
base = alt.Chart(projected_data)

# data scatter
scatter = base.mark_point(opacity = 0.2).encode(
    x = alt.X('PC1:Q', title = 'Temperature PC'),
    y = alt.Y('PC2:Q', title = 'Income and educational expenditure PC'),
)

# show
scatter


# In[110]:


# find cutoff value
sum = projected_data.PC1 + projected_data.PC2
pc_2plus4 = sorted(sum, reverse = True)

#print
pc_2plus4[0:15]


# In[112]:


# store outlying rows using cutoff
outliers = projected_data[(projected_data.PC1 + projected_data.PC2) > 7]

# plot outliers in red
pts = alt.Chart(outliers).mark_circle(
    color = 'red',
    opacity = 0.6
).encode(
    x = 'PC1',
    y = 'PC2'
)

# layer
scatter + pts


# In[82]:


y = sonGDIMavg.MEANc.values
x_df = sonGDIMavg[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)


# In[83]:


y = allGDIMavg.MEANc.values
x_df = allGDIMavg[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)


# In[84]:


y = daughterGDIMmom.MEANc.values
x_df = daughterGDIMmom[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)


# In[85]:


y = daughterGDIMdad.MEANc.values
x_df = daughterGDIMdad[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)


# In[86]:


y = sonGDIMdad.MEANc.values
x_df = sonGDIMdad[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)


# In[87]:


y = sonGDIMmom.MEANc.values
x_df = sonGDIMmom[['MEANp','GEI %','Gender Economic Participation and Opportunity Equality %','Political Instability Estimate','IQ','Average Income (USD)','Education expenditure\nper inhabitant (USD)','Daily maximum\ntemperature Celsius']]
x_df.head()

# add column of ones (for intercept)
x_mx = add_dummy_feature(x_df, value = 1)
# fit first model
mlr = lm.LinearRegression(fit_intercept = False)
mlr.fit(x_mx, y)
mlr.coef_

# fitted values and residuals
fitted = mlr.predict(x_mx)
resid = y - fitted
# error variance estimate
n, p = x_mx.shape
sigmasqhat = resid.var()*(n - 1)/(n - p)
sigmasqhat

xtx = x_mx.transpose().dot(x_mx) # X'X
xtxinv = np.linalg.inv(xtx) # (X'X)^{-1}
vhat = sigmasqhat*xtxinv # V
vhat

coef_se = np.sqrt(vhat.diagonal()) # (v_jj)^{1/2}
coef_se

# using sklearn.metrics.r2_score
r2_score(y, fitted)

