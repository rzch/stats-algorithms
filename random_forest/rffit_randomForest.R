rm(list = ls()) #Clear environment variables
dev.off() #Close plots
cat("\014") #Clear console

library(randomForest)

#Predicting whether a match came from the WTA or the ATP tour from match stats
#This script learns a random forest classifier using the randomForest package

df <- read.csv("tour_dataset.csv")
df$is_wta <- as.factor(df$is_wta)

#sample training data
set.seed (2)
train <- sample(1:nrow(df), round(0.75*nrow(df)))

df_train <- df[train,]
df_test <- df[-train,]

rffit = randomForest(is_wta~w_ace+w_df+w_1stPct+w_1stPctWon+w_2ndPctWon+l_ace+l_df+l_1stPct+l_1stPctWon+l_2ndPctWon, 
                    data=df_train, mtry=10, importance =TRUE)

print(rffit)
summary(rffit)
plot(rffit)

is_wta_pred = predict(rffit , df_test, type="class") #make predictions on training data
confMat <- table(df_test$is_wta, is_wta_pred) #compute confusion matrix
accuracy <- sum(diag(confMat))/sum(confMat) #compute training prediction accuracy

#compute precision, recall and F-scores
precision1 <- confMat[2,2]/(sum(confMat[,2]))
recall1 <- confMat[2,2]/(sum(confMat[2,]))
f.score1 <- 2*precision1*recall1/(precision1 + recall1)
precision0 <- confMat[1,1]/(sum(confMat[,1]))
recall0 <- confMat[1,1]/(sum(confMat[1,]))
f.score0 <- 2*precision0*recall0/(precision0 + recall0)