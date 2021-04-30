# Load Salary_Data.csv dataset
library(readr)
sa_da <- read_csv("E:\\Assignment\\4) Simple linear regression\\Salary_Data.csv")
View(sa_da)

# Exploratory data analysis
summary(sa_da)

#Scatter plot
plot(sa_da$YearsExperience, sa_da$Salary)  # plot(X,Y)

?plot

attach(sa_da)


#Correlation Coefficient (r)
cor(YearsExperience, Salary)             # cor(X,Y)

# Simple Linear Regression model
reg <- lm(Salary ~ YearsExperience) # lm(Y ~ X)

summary(reg)

pred <- predict(reg)

reg$residuals
sum(reg$residuals)

mean(reg$residuals)
sqrt(sum(reg$residuals^2)/nrow(sa_da))  #RMSE

sqrt(mean(reg$residuals^2))

confint(reg,level=0.95)
predict(reg,interval="predict")

# ggplot for adding regresion line for data
library(ggplot2)

?ggplot2

ggplot(data = sa_da, aes(x = YearsExperience, y = Salary)) + 
  geom_point(color='blue') +
  geom_line(color='red',data = sa_da, aes(x=YearsExperience, y=pred))

?ggplot2

########################
# A simple ggplot code for directly showing the line

# ggplot(sa_da,aes(YearsExperiance,Salary))+stat_summary(fun.data=mean_cl_normal) + geom_smooth(method='lm')

####################

# Logrithamic Model

# x = log(YearsExperiance); y = Salary

plot(log(YearsExperience), Salary)
cor(log(YearsExperience), Salary)

reg_log <- lm(Salary ~ log(YearsExperience))   # lm(Y ~ X)

summary(reg_log)
predict(reg_log)

reg_log$residuals
sqrt(sum(reg_log$residuals^2)/nrow(sa_da))  #RMSE

confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

######################

# Exponential Model

# x = YearsExperiance and y = log(Salary)

plot(YearsExperience, log(Salary))

cor(YearsExperience, log(Salary))

reg_exp <- lm(log(Salary) ~ YearsExperience)  #lm(log(Y) ~ X)

summary(reg_exp)

reg_exp$residuals

sqrt(mean(reg_exp$residuals^2))

logat <- predict(reg_exp)
at <- exp(logat)

error = sa_da$Salary - at
error

sqrt(sum(error^2)/nrow(sa_da))  #RMSE

confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")

##############################
# Polynomial model with 2 degree (quadratic model)

plot(YearsExperience, Salary)
plot(YearsExperience*YearsExperience, Salary)

cor(YearsExperience*YearsExperience, Salary)

plot(YearsExperience*YearsExperience, log(Salary))

cor(YearsExperience, log(Salary))
cor(YearsExperience*YearsExperience, log(Salary))

# lm(Y ~ X + I(X*X) +...+ I(X*X*X...))

reg2degree <- lm(log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience))

summary(reg2degree)

logpol <- predict(reg2degree)
expy <- exp(logpol)

err = sa_da$Salary - expy

sqrt(sum(err^2)/nrow(sa_da))  #RMSE

confint(reg2degree,level=0.95)
predict(reg2degree,interval="confidence")

# visualization
ggplot(data = sa_da, aes(x = YearsExperience + I(YearsExperience^2), y = log(Salary))) + 
  geom_point(color='blue') +
  geom_line(color='red',data = sa_da, aes(x=YearsExperience+I(YearsExperience^2), y=logpol))


##############################
#  Polynomial model with 3 degree

reg3degree<-lm(log(Salary)~YearsExperience + I(YearsExperience*YearsExperience) + I(YearsExperience*YearsExperience*YearsExperience))

summary(reg3degree)
logpol3 <- predict(reg3degree)
expy3 <- exp(logpol3)


# visualization
ggplot(data = sa_da, aes(x = YearsExperience + I(YearsExperience^2) + I(YearsExperience^3), y = Salary)) + 
  geom_point(color='blue') +
  geom_line(color='red',data = sa_da, aes(x=YearsExperience+I(YearsExperience^2)+I(YearsExperience^3), y=expy3))

################################
