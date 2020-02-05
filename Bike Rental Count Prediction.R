rm(list = ls(all = T))
setwd("C:/Users/GAURAV/Desktop/Edwisor/Projects/Bike Rental Count Prediction")
getwd()

df = read.csv("day.csv")

################# Exploratory Data Analysis #####################

head(df)

str(df)

summary(df)

# Dropping few variables as they do not contain any useful information.
df[c('casual', 'registered', 'dteday', 'instant')] = NULL

continuous_columns = c('temp', 'atemp', 'hum', 'windspeed', 'cnt')
categorical_columns = c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit')

################ Missing Value Analysis #########################

sum(is.na(df))

# There are no missing values in the dataset.

################ Outlier Analysis #######################

for (column in continuous_columns) {
  boxplot(df[column], xlab=column)
}

# We can see inliers in humidity and outliers in windspeed.
# Imputing these values with median.
  
q25 = quantile(df$hum, probs = c(0.25))
q75 = quantile(df$hum, probs = c(0.75))
iqr = q75 - q25
  
minimum = q25 - (1.5 * iqr)
maximum = q75 + (1.5 * iqr)
  
df$hum[df$hum < minimum] = median(df$hum)

q25 = quantile(df$windspeed, probs = c(0.25))
q75 = quantile(df$windspeed, probs = c(0.75))
iqr = q75 - q25

minimum = q25 - (1.5 * iqr)
maximum = q75 + (1.5 * iqr)

df$windspeed[df$windspeed > maximum] = median(df$windspeed)

boxplot(df$hum)
boxplot(df$windspeed)

################# Visualization ##################

# Univariate Analysis

plot(density(df$temp), xlab = 'Temp')
plot(density(df$atemp), xlab = 'aTemp')
plot(density(df$hum), xlab = 'Humidity')
plot(density(df$windspeed), xlab = 'Windspeed')
plot(density(df$cnt), xlab = 'Count')

# Bivariate Analysis

plot(df$cnt, df$temp, xlab = "Count", ylab = "Temperature")
plot(df$cnt, df$atemp, xlab = "Count", ylab = "Feels like temperature")
plot(df$cnt, df$hum, xlab = "Count", ylab = "Humidity")
plot(df$cnt, df$windspeed, xlab = "Count", ylab = "Windspeed")

# From above plots we can see that temperature(temp) and feels like temperature(atemp) 
# are directly proportional to bike rental count.
# As temp or atemp variable increases cnt variable also increases.
# Humidity and windspeed do not impact bike count.

plot(df$season, df$cnt, xlab = "Season", ylab = "Count")
plot(df$yr, df$cnt, xlab = "Year", ylab = "Count")
plot(df$mnth, df$cnt, xlab = "Month", ylab = "Count")
plot(df$holiday, df$cnt, xlab = "Holiday", ylab = "Count")
plot(df$weekday, df$cnt, xlab = "Weekday", ylab = "Count")
plot(df$workingday, df$cnt, xlab = "Working Day", ylab = "Count")
plot(df$weathersit, df$cnt, xlab = "Weather", ylab = "Count")

# Based on the plots,
# Bike rental count is high in season 3 which is fall and low in season 1 which is spring.
# Bike rental count is high in the year 1 which is 2012.
# Bike rental count is high in 8 which is in august and low in 1 which is in january.
# Bike rental count is high in 0 which is holiday and low in 1 which is working day.
# Bike rental count is high in 5 which is friday and low in 0 which is sunday.
# Bike rental count is high in 1 which is working day and low in 0 which is holiday.
# Bike rental count is higher in 1 which clear,few clouds,partly cloudy and there is no bikes rental in 4.

######################### Feature Selection #############################

# For continuous values, we will be using correlation matrix.

corr = cor(df[continuous_columns])

library(corrplot)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(corr, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45, 
         method = "color", col=col(200), addCoef.col = "black")

# From the above plot, we can see temp and atemp are highly correlated to each other.
# Hence, we need to remove any one variable.
# We select 'atemp' to be dropped.

df$atemp = NULL

# For categorical variables, we will be doing ANOVA test.

summary(aov(cnt~season, data = df))
summary(aov(cnt~yr, data = df))
summary(aov(cnt~mnth, data = df))
summary(aov(cnt~holiday, data = df))
summary(aov(cnt~weekday, data = df))
summary(aov(cnt~workingday, data = df))
summary(aov(cnt~weathersit, data = df))

# According to ANOVA test, columns 'holiday','weekday','workingday' have p value greater than 0.05.
# Hence, dropping those variables as well.

df$holiday = NULL
df$weekday = NULL
df$workingday = NULL

str(df)

########################## Feature Scaling ###############################
qqnorm(df$temp)
qqnorm(df$hum)
qqnorm(df$windspeed)
qqnorm(df$cnt)

# Based on the above plots, the data is normalized.

######################### Model Development #################################

set.seed(42)
index = sample(1:nrow(df), 0.8*nrow(df))

X = df[index,]
y = df[-index,]

# Linear Regression
lm_model = lm(cnt ~., data = df)
summary(lm_model)

y_pred = predict(lm_model, y[,-8])
rmse_LR = sqrt(mean((y$cnt-y_pred)^2))
rmse_LR
r_squared_LR = summary(lm_model)$r.squared
r_squared_LR

####### Decision Tree #########
require(tree)
dt_model = tree(cnt~., data = df)
summary(dt_model)

predictions_DT = predict(dt_model, y[,-8])
rmse_DT = sqrt(mean((y$cnt-predictions_DT)^2))
rmse_DT
r_squared_DT = 1 - sum((y$cnt-predictions_DT)^2)/sum((y$cnt-mean(y$cnt))^2)
r_squared_DT

####### Random Forest #########
library(randomForest)
rf_model = randomForest(cnt~., data = df, importance=TRUE, ntree=100)

predictions_RF = predict(rf_model, y[,-8])
rmse_RF = sqrt(mean((y$cnt-predictions_RF)^2))
rmse_RF
r_squared_RF = 1 - sum((y$cnt-predictions_RF)^2)/sum((y$cnt-mean(y$cnt))^2)
r_squared_RF

####### Gradient Boosting ###########
library(gbm)
gb_model = gbm(cnt~., data = df, n.trees = 500, interaction.depth = 2)

predictions_GB = predict(gb_model, y[,-8], n.trees = 500)
rmse_GB = sqrt(mean((y$cnt-predictions_GB)^2))
rmse_GB
r_squared_GB = 1 - sum((y$cnt-predictions_GB)^2)/sum((y$cnt-mean(y$cnt))^2)
r_squared_GB

########## Summary of all models #############
models = c("Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting")
RMSE = c(rmse_LR, rmse_DT, rmse_RF, rmse_GB)
r2 = c(r_squared_LR, r_squared_DT, r_squared_RF, r_squared_GB)

results = data.frame(models, RMSE, r2)
results

########## Best Model Selection ################

#From the above results we can see random forest gives the best output.

true_values = y
predicted_values = predict(rf_model, y[,-8])
output = data.frame(true_values, predicted_values)
write.csv(output, "output_R.csv")
