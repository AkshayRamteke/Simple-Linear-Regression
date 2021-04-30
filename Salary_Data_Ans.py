# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
SaDa=pd.read_csv("E:\\Assignment\\4) Simple linear regression\\Salary_Data.csv")
SaDa.columns

plt.hist(SaDa.YearsExperience)
plt.boxplot(SaDa.YearsExperience,0,"rs",0)


plt.hist(SaDa.Salary)
plt.boxplot(SaDa.Salary)

plt.plot(SaDa.YearsExperience,SaDa.Salary,"bo");plt.xlabel("YearsExperience");plt.ylabel("Salary")


SaDa.Salary.corr(SaDa.YearsExperience) # # correlation value between X and Y
np.corrcoef(SaDa.Salary,SaDa.YearsExperience)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=SaDa).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(SaDa.iloc[:,0]) # Predicted values of Salary using the model

# Visualization of regresion line over the scatter plot of YearsExperience and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=SaDa['YearsExperience'],y=SaDa['Salary'],color='red');plt.plot(SaDa['YearsExperience'],pred,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')

pred.corr(SaDa.Salary) # 0.97

# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=SaDa).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(SaDa['YearsExperience']))
pred2.corr(SaDa.Salary)
# pred2 = model2.predict(SaDa.iloc[:,0])
pred2
plt.scatter(x=SaDa['YearsExperience'],y=SaDa['Salary'],color='green');plt.plot(SaDa['YearsExperience'],pred2,color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')

# Exponential transformation
model3 = smf.ols('np.log(Salary)~YearsExperience',data=SaDa).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(SaDa['YearsExperience']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(Salary) in preparing model so we need to convert it back
pred3
pred3.corr(SaDa.Salary)
plt.scatter(x=SaDa['YearsExperience'],y=SaDa['Salary'],color='green');plt.plot(SaDa.YearsExperience,np.exp(pred_log),color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')
resid_3 = pred3-SaDa.Salary
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=SaDa.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")



# Quadratic model
SaDa["YearsExperience_Sq"] = SaDa.YearsExperience*SaDa.YearsExperience
model_quad = smf.ols("Salary~YearsExperience+YearsExperience_Sq",data=SaDa).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(SaDa)

model_quad.conf_int(0.05) # 
plt.scatter(SaDa.YearsExperience,SaDa.Salary,c="b");plt.plot(SaDa.YearsExperience,pred_quad,"r")

plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 

## End 

