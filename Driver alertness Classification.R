#installing packages

install.packages("rpart")
install.packages("rpart.plot")
install.packages("ROCR")
install.packages("e1071")
install.packages("randomForest")
install.packages("caTools")
install.packages("caret")
install.packages("MASS")
install.packages("ggplot2")
install.packages("ggcorrplot")
install.packages("dplyr")
install.packages("RColorBrewer")
install.packages("heatmap.plus")
install.packages("data.table")

#calling out libraries
library(rpart)
library(rpart.plot)
library(ROCR)
library(e1071)
library(randomForest)
library(caTools)
library(caret)
library(MASS)
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(RColorBrewer)
library(heatmap.plus)
library(data.table)

#reading the dataset into R


#Seeing the structure of the dataset

str(fordData)


# converting into factor
fordTrain$IsAlert <- as.factor(fordTrain$IsAlert)
fordTrain$E3 <- as.factor(fordTrain$E3)
fordTrain$E9 <- as.factor(fordTrain$E9)
#fordTrain$E7 <- as.factor(fordTrain$E7)
fordTrain$E8 <- as.factor(fordTrain$E8)
fordTrain$V5 <- as.factor(fordTrain$V5)
fordTrain$V10 <- as.factor(fordTrain$V10)

#splitting the data into 70% and 30% 
fordTrainSet <- subset(fordTrain,fordTrain$TrialID <= 357)
fordTestSet <- subset(fordTrain,fordTrain$TrialID>357)


#barchart for difference between means of the classes 0 and 1
subset.alert <- subset.data.frame(fordData,fordData$IsAlert==1)
subset.notalert <- subset.data.frame(fordData,fordData$IsAlert==0)
subset.alert.scaled <- as.data.frame(scale(subset.alert[,-c(1,2,3,14,20,27,19,32)]))
subset.notalert.scaled <- as.data.frame(scale(subset.notalert[,-c(1,2,3,14,20,27,19,32)]))
colmeans_alert<- as.vector(colMeans(subset.alert.scaled))
names(colmeans_alert) <- names(subset.alert.scaled)
colmeans_notalert <- as.vector(colMeans(subset.notalert.scaled))
names(colmeans_notalert) <- names(subset.notalert.scaled)
diffmeans <- abs(colmeans_alert - colmeans_notalert)
diffmeans <- sort(diffmeans,decreasing = TRUE)
barchart(diffmeans)

#barchart for difference between variances of the classes 0 and 1

variance_alert <- colVars(as.matrix(subset.alert.scaled))
names(variance_alert) <- names(subset.alert.scaled)
variance_notalert <- colVars(as.matrix(subset.notalert.scaled))
names(variance_notalert) <- names(subset.notalert.scaled)
diffvar <- abs(variance_alert - variance_notalert)
diffvar <- sort(diffvar,decreasing = TRUE)
barchart(diffvar)

#generating a heatmap to see the relationship between variables 
temp <- fordData[,-c(11,29,31,1,2,3,14,20,27,19,32)]

colors = c( seq(-3,-2,length=100), seq(-2,0.5,length=100), seq(0.5,6,length=100))
my_palette <- colorRampPalette(c("red", "white","blue"))
heatmap.2(cor(temp),col=my_palette,Colv =NA,Rowv = NA,dendrogram = "none",trace="none",cellnote = round(cor(temp),1),symm=F,symkey=F,symbreaks=T, scale="none")

#Relationship between the variables P3 and P4 

graph <- fordData %>% ggplot(aes(fordData$P3,fordData$P4))


#relationship between variables P3 and P4 after applying log tranformation 

graph+geom_point()+scale_x_log10()+scale_y_log10() +ggtitle("Relation between P3 and P4 variable") +xlab("P4")+ylab("P3")

#making the dropped columns null
fordTrain$ObsNum=NULL
fordTrain$P2=NULL 
fordTrain$P8=NULL
fordTrain$V7=NULL
fordTrain$V9=NULL
fordTrain$V8=NULL
fordTrain$P3=NULL

#logistic regression 


fordLog<-  glm(IsAlert ~ .,data = fordTrainSet,family = binomial)
summary(fordLog)
predtest <- predict(fordLog,newdata = fordTestSet,type="response")
table(fordTestSet$IsAlert,predtest>0.5)
predglm  <- prediction(predtest,fordTestSet$IsAlert)
perfglm<- performance(predglm,"tpr","fpr")
#area under curve
as.numeric(performance(predglm, "auc")@y.values)
plot(perfglm) +abline(v=0.36)

#classification Tree


fordCART <-  rpart(IsAlert ~.,data = fordTrainSet,method="class")
prp(fordCART)
predFordTree <- predict(fordCART,newdata = fordTestSet,type = "class")
#confusion matrix for classification tree
table(fordTestSet$IsAlert,predFordTree)


#generate ROC and AUC


predROC <- predict(fordCART,newdata = fordTestSet)
pred  <- prediction(predROC[,2],fordTestSet$IsAlert)
perf<- performance(pred,"tpr","fpr")


#ROC for classification tree


as.numeric(performance(pred, "auc")@y.values)
plot(perf) +abline(v=0.45)

#cross validation


set.seed(2)
numFolds <- trainControl(method = "cv",number=20)
cartGrid = expand.grid( .cp = seq(0.002,0.1,0.002))
train(IsAlert~.,data=fordTrainSet,method="rpart",trControl=numFolds,tuneGrid=cartGrid)
fordTree <- rpart(IsAlert~.,data=fordTrainSet,method="class",cp=0.002)
prp(fordTree)
#ROC for cross validation

predAlertCP <- predict(fordTree,newdata =fordTestSet,type="class")
predCpROC <- predict(fordTree,newdata = fordTestSet)
predCp  <- prediction(predCpROC[,2],fordTestSet$IsAlert)
perfCp<- performance(predCp,"tpr","fpr")
as.numeric(performance(predCp, "auc")@y.values)
plot(perfCp) +abline(v=0.34)


#confusion matrix for cross validation

table(fordTestSet$IsAlert,predAlertCP)


#random forest


set.seed(1000)
RFModel<-randomForest(IsAlert ~ ., data=fordTrainSet,ntree=300,nodesize=15)
predRF <-  predict(RFModel,newdata = fordTestSet)
# confusion matrix for random forest

table(fordTestSet$IsAlert,predRF)


# Plotting a graph to get to know about the most important variables for rf
vu = varUsed(RFModel, count=TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE) 
vu = varUsed(RFModel, count=TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(RFModel$forest$xlevels[vusorted$ix]))
varImpPlot(RFModel)


#LDA

fordLda <- MASS::lda(IsAlert ~ .,data = fordTrainSet)
fordLdaPred <- predict(fordLda,newdata = fordTestSet)
summary(fordLda)


# confusion matrix for LDA

table(fordTestSet$IsAlert,fordLdaPred$posterior[,2]>0.5)


#ROC for LDA

predLdaROC  <- prediction(fordLdaPred$posterior[,2],fordTestSet$IsAlert)
perfLda<- performance(predLdaROC,"tpr","fpr")
as.numeric(performance(predLdaROC, "auc")@y.values)
plot(perfLda) +abline(v=0.5)
