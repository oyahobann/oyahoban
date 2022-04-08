library(caret)
library(randomForest)
library(caTools)
library(gbm)
library(data.table)
library(ISLR)
library(boot)
require(jsonlite)
require(httr)
library(lubridate)
library(dplyr)
library(ggplot2)
library(knitr)
library(tidyr)
library(tidyverse)
library(scales)
library(ggcorrplot)
library(forecast)
library(urca)
library(zoo)
library(reshape)
library(GGally)
library(PerformanceAnalytics)
library(ROCR)
library(gbm)
library("Metrics")
library("pROC")

set.seed(425)

org_train_data <- read.csv("C:/Users/DELL/Desktop/2020-2021 Ders 2/IE 425/Proje/kobe-train.csv", header = TRUE,sep = ",")
org_test_data <-  read.csv("C:/Users/DELL/Desktop/2020-2021 Ders 2/IE 425/Proje/kobe-test.csv", header = TRUE,sep = ",")

train_data <- read.csv("C:/Users/DELL/Desktop/2020-2021 Ders 2/IE 425/Proje/kobe-train.csv", header = TRUE,sep = ",")

setDT(train_data)
setDT(org_train_data)
setDT(org_test_data)

str(train_data)

train_data[,loc:= sqrt(loc_x^2 + loc_y^2)]
train_data <- train_data[,-c("loc_x","loc_y")]




train_data[grepl("@", train_data$matchup, fixed=TRUE)==TRUE, ishome:=1]
train_data[is.na(train_data$ishome)==TRUE, ishome:=0]
str(train_data)

train_data <- train_data[,-c("matchup")]

train_data[,time_remaining := (minutes_remaining*60 + seconds_remaining)]
train_data <- train_data[,-c("minutes_remaining","seconds_remaining")]
train_data <- train_data[,-c("team_name","team_id")]




train_data[,shot_made_flag:=as.factor(train_data$shot_made_flag)]


str(train_data)
train_data[,action_type:=as.factor(train_data$action_type)]
train_data[,season:=as.factor(train_data$season)]
train_data[,shot_type:=as.factor(train_data$shot_type)]
train_data[,shot_zone_area:=as.factor(train_data$shot_zone_area)]
train_data[,shot_zone_basic:=as.factor(train_data$shot_zone_basic)]
train_data[,shot_zone_range:=as.factor(train_data$shot_zone_range)]

train_data[,game_date:=as.factor(train_data$game_date)]

train_data[,opponent:=as.factor(train_data$opponent)]
train_data[,combined_shot_type:=as.factor(train_data$combined_shot_type)]




str(train_data)

split=sample.split(train_data$shot_made_flag,SplitRatio=0.7)




test_data=subset(train_data,split==FALSE)
train_data=subset(train_data,split==TRUE)





str(train_data)
train_data <- train_data[,-c("X")]

##########################################################################################################

df <- data.frame(train_data$game_date,train_data$season)
df

train_data <- train_data[,-c("game_date")]

#as we can see on the table season is more aggregated version of game date, 
#so we can discard game date and use season for shorter run time of the model.
####################################################################################################
#Prediction with rpart

ctrl1=trainControl(method='cv',number=10)
fit1=train(shot_made_flag~., data= train_data, method = "rpart",
           trControl = ctrl1, tuneGrid = expand.grid(cp=(1:10)*0.001))
#best cp = 0.002

fit1

# Accuracy value = 0.6762593
pred1 <- predict(fit1, newdata=test_data,type="raw")


predictions1= as.numeric(pred1)

auc(roc(test_data$shot_made_flag, predictions1))

#AUC value = 0.6643

confusionMatrix(as.factor(pred1) ,as.factor(test_data$shot_made_flag), positive="1")

pred1 <- prediction(predictions1,test_data$shot_made_flag)
perf=performance(pred1,"tpr","fpr")
plot(perf)
as.numeric(performance(pred1,"auc")@y.values)




####################################################################################################


#Prediction with GBM

ctrl2 = trainControl(method = "cv", number = 10)

gbmGrid1=expand.grid(interaction.depth = c(3, 4), 
                     n.trees = c(50, 60,100,150), 
                     shrinkage = (1:2)*0.1,
                     n.minobsinnode = c(10, 20))


fit2=train(shot_made_flag~., data=train_data, method="gbm", metric='Accuracy',
           trControl = ctrl2,tuneGrid = gbmGrid1)
fit2


#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1
#and n.minobsinnode = 20.
#Accuracy = 0.6800278

pred2 <- predict(fit2, newdata=test_data,type="raw")


predictions2= as.numeric(pred2)

auc(roc(test_data$shot_made_flag, predictions2))

#AUC = 0.6666

confusionMatrix(as.factor(pred2) ,as.factor(test_data$shot_made_flag), positive="1")

pred2 <- prediction(predictions2,test_data$shot_made_flag)
perf2=performance(pred2,"tpr","fpr")
plot(perf2)
as.numeric(performance(pred2,"auc")@y.values)


##########################################################################################################
# GLM Prediction Part

fit3=glm(shot_made_flag~.,data=train_data,family=binomial(link = "logit"))

fit3

test_datav2 <- test_data[action_type != "Driving Floating Bank Jump Shot",]
test_datav2 <- test_datav2[action_type != "Running Slam Dunk Shot",]
test_datav2 <- test_datav2[action_type != "Running Tip Shot",]
test_datav2 <- test_datav2[action_type != "Turnaround Finger Roll Shot",]
test_datav2 <- test_datav2[game_id != "29600095",]
test_datav2 <- test_datav2[game_id != "29600362",]
test_datav2 <- test_datav2[game_id != "29600382",]


pred3 <- predict(fit3, newdata=test_datav2,type="response")


predictions3= as.numeric(pred3)

auc(roc(test_datav2$shot_made_flag, predictions3))

#AUC = 0.6639954



pred3 <- prediction(predictions3,test_datav2$shot_made_flag)
perf3=performance(pred3,"tpr","fpr")
plot(perf3)
as.numeric(performance(pred3,"auc")@y.values)




############################################################################################################

#Prediction with RF

fit4=train(shot_made_flag~., data=train_data, method = "rf",metric="Accuracy",
           trControl = ctrl1, tuneGrid = expand.grid(mtry=(1:9)), ntree=400)


fit4




###########################################################################################################
#GLM with Caret Package


ctrl2 = trainControl(method = "cv", number = 10)




fit5=train(shot_made_flag~., data=train_data, method="glm", metric='Accuracy',
           trControl = ctrl2)
fit5



##########################################################################################################




##########################################################################################


predictions= as.numeric(pred)

auc(roc(test_data$shot_made_flag, pred))

table(test_data,pred>=0.5)

levels(pred)
levels(test_data$shot_made_flag)

confusionMatrix(data=pred,test_data$shot_made_flag,  positive="1")

perf=performance(pred,"tpr","fpr")
plot(perf)
as.numeric(performance(pred,"auc")@y.values)





#install.packages('e1071', dependencies=TRUE)
################################################################################################


org_train_data[,loc:= sqrt(loc_x^2 + loc_y^2)]
org_train_data <- org_train_data[,-c("loc_x","loc_y")]
org_train_data[grepl("@", org_train_data$matchup, fixed=TRUE)==TRUE, ishome:=1]
org_train_data[is.na(org_train_data$ishome)==TRUE, ishome:=0]
org_train_data <- org_train_data[,-c("matchup")]
org_train_data[,time_remaining := (minutes_remaining*60 + seconds_remaining)]
org_train_data <- org_train_data[,-c("minutes_remaining","seconds_remaining")]
org_train_data <- org_train_data[,-c("team_name","team_id")]
org_train_data[,shot_made_flag:=as.factor(org_train_data$shot_made_flag)]
org_train_data[,action_type:=as.factor(org_train_data$action_type)]
org_train_data[,season:=as.factor(org_train_data$season)]
org_train_data[,shot_type:=as.factor(org_train_data$shot_type)]
org_train_data[,shot_zone_area:=as.factor(org_train_data$shot_zone_area)]
org_train_data[,shot_zone_basic:=as.factor(org_train_data$shot_zone_basic)]
org_train_data[,shot_zone_range:=as.factor(org_train_data$shot_zone_range)]
org_train_data[,game_date:=as.factor(org_train_data$game_date)]
org_train_data[,opponent:=as.factor(org_train_data$opponent)]
org_train_data[,combined_shot_type:=as.factor(org_train_data$combined_shot_type)]

org_train_data <- org_train_data[,-c("X")]
org_train_data <- org_train_data[,-c("game_date")]


org_test_data[,loc:= sqrt(loc_x^2 + loc_y^2)]
org_test_data <- org_test_data[,-c("loc_x","loc_y")]
org_test_data[grepl("@", org_test_data$matchup, fixed=TRUE)==TRUE, ishome:=1]
org_test_data[is.na(org_test_data$ishome)==TRUE, ishome:=0]
org_test_data <- org_test_data[,-c("matchup")]
org_test_data[,time_remaining := (minutes_remaining*60 + seconds_remaining)]
org_test_data <- org_test_data[,-c("minutes_remaining","seconds_remaining")]
org_test_data <- org_test_data[,-c("team_name","team_id")]
org_test_data[,shot_made_flag:=as.factor(org_test_data$shot_made_flag)]
org_test_data[,action_type:=as.factor(org_test_data$action_type)]
org_test_data[,season:=as.factor(org_test_data$season)]
org_test_data[,shot_type:=as.factor(org_test_data$shot_type)]
org_test_data[,shot_zone_area:=as.factor(org_test_data$shot_zone_area)]
org_test_data[,shot_zone_basic:=as.factor(org_test_data$shot_zone_basic)]
org_test_data[,shot_zone_range:=as.factor(org_test_data$shot_zone_range)]
org_test_data[,game_date:=as.factor(org_test_data$game_date)]
org_test_data[,opponent:=as.factor(org_test_data$opponent)]
org_test_data[,combined_shot_type:=as.factor(org_test_data$combined_shot_type)]
org_test_data <- org_test_data[,-c("game_date")]









##############################################################################################3
#Final Prediction with GBM


ctrl2 = trainControl(method = "cv", number = 10)

gbmGrid1=expand.grid(interaction.depth = c(3, 4), 
                     n.trees = c(50,60,100), 
                     shrinkage = (1:2)*0.1,
                     n.minobsinnode = c(10, 20))



gbmGrid2=expand.grid(interaction.depth = c(4), 
                     n.trees = c(50), 
                     shrinkage = (2)*0.1,
                     n.minobsinnode = c(10))



fitfinal=train(shot_made_flag~., data=org_train_data, method="gbm", metric='Accuracy',
           trControl = ctrl2,tuneGrid = gbmGrid1)

fitfinal


#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 100, interaction.depth = 4, shrinkage = 0.1
#and n.minobsinnode = 10.

  
fitfinal2=train(shot_made_flag~., data=org_train_data, method="gbm", metric='Accuracy',
               trControl = ctrl2,tuneGrid = gbmGrid2)
fitfinal2


#The final values used for the model were n.trees = 50, interaction.depth = 4, shrinkage = 0.2
#and n.minobsinnode = 10.


#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 60, interaction.depth = 4, shrinkage = 0.2
#and n.minobsinnode = 20. Accracy = 0.6790396



pred_final2 <- predict(fitfinal, newdata=org_test_data,type="prob")


pred_final2 <- predict(fitfinal, org_test_data, type= "prob")
shot_made_flag<-pred_final2[,2]
shot_id=org_test_data$X.1
submission2= data.table(shot_id,shot_made_flag)
write.csv(submission2,"C:/Users/DELL/Desktop/submission2.csv",row.names = F)

#################################################################3



gbmGrid3=expand.grid(interaction.depth = c(3), 
                     n.trees = c(150), 
                     shrinkage = (1)*0.1,
                     n.minobsinnode = c(20))



fitfinal3=train(shot_made_flag~., data=org_train_data, method="gbm", metric='Accuracy',
               trControl = ctrl2,tuneGrid = gbmGrid3)

fitfinal3

pred_final3 <- predict(fitfinal3, newdata=org_test_data,type="prob")


pred_final3 <- predict(fitfinal3, org_test_data, type= "prob")
shot_made_flag<-pred_final3[,2]
shot_id=org_test_data$X.1
submission3= data.table(shot_id,shot_made_flag)
write.csv(submission3,"C:/Users/DELL/Desktop/submission3.csv",row.names = F)

