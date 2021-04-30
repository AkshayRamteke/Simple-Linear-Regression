# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
delt=pd.read_csv("E:\\Assignment\\4) Simple linear regression\\delivery_time.csv")
delt.columns
delt=delt.rename(columns={'Delivery Time':'DeliveryTime'})
delt=delt.rename(columns={'Sorting Time':'SortingTime'})


plt.hist(delt.DeliveryTime)
plt.boxplot(delt.DeliveryTime,0,"rs",0)


plt.hist(delt.SortingTime)
plt.boxplot(delt.SortingTime)

plt.plot(delt.DeliveryTime,delt.SortingTime,"bo");plt.xlabel("DeliveryTime");plt.ylabel("SortingTime")

delt.corr()
delt.SortingTime.corr(delt.DeliveryTime) # # correlation value between X and Y
np.corrcoef(delt.SortingTime,delt.DeliveryTime)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("SortingTime~DeliveryTime",data=delt).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(delt.iloc[:,0]) # Predicted values of Sorting Time using the model



# Visualization of regresion line over the scatter plot of Delivery Time and Sorting Time
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=delt['DeliveryTime'],y=delt['SortingTime'],color='red');plt.plot(delt['DeliveryTime'],pred,color='black');plt.xlabel('DeliveryTime');plt.ylabel('SortingTime')

pred.corr(delt.SortingTime) # 0.82

# Transforming variables for accuracy
model2 = smf.ols('SortingTime~np.log(DeliveryTime)',data=delt).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(delt['DeliveryTime']))
pred2.corr(delt.SortingTime)
# pred2 = model2.predict(delt.iloc[:,0])
pred2
plt.scatter(x=delt['DeliveryTime'],y=delt['SortingTime'],color='green');plt.plot(delt['DeliveryTime'],pred2,color='blue');plt.xlabel('DeliveryTime');plt.ylabel('SortingTime')

# Exponential transformation
model3 = smf.ols('np.log(SortingTime)~DeliveryTime',data=delt).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(delt['DeliveryTime']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(SortingTime) in preparing model so we need to convert it back
pred3
pred3.corr(delt.SortingTime)
plt.scatter(x=delt['DeliveryTime'],y=delt['SortingTime'],color='green');plt.plot(delt.DeliveryTime,np.exp(pred_log),color='blue');plt.xlabel('DeliveryTime');plt.ylabel('SortingTime')
resid_3 = pred3-delt.SortingTime
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=delt.SortingTime);plt.xlabel("Predicted");plt.ylabel("Actual")



# Quadratic model
delt["DeliveryTime_Sq"] = delt.DeliveryTime*delt.DeliveryTime
model_quad = smf.ols("SortingTime~DeliveryTime+DeliveryTime_Sq",data=delt).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(delt)

model_quad.conf_int(0.05) 
plt.scatter(delt.DeliveryTime,delt.SortingTime,c="b");plt.plot(delt.DeliveryTime,pred_quad,"r")

plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 






