args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])

decTree_accuracySum<-0
svm_accuracySum<-0
naive_accuracySum<-0
knn_accuracySum<-0
lr_accuracySum<-0
neuralnet_accuracySum<-0
bagging_accuracySum<-0
randomForest_accuracySum <- 0
boosting_accuracySum<-0
nn_accuracySum<-0
options(warn=0)
set.seed(123)
# create 10 samples

library(rpart)
library(class)
library(e1071)
library(adabag)
library(randomForest)
library(neuralnet)

for(i in 1:10) {
  cat("Running sample ",i,"\n")
  #dataURL<-'http://www.utdallas.edu/~axn112530/cs6375/creditset.csv'
  #header<-T
  d<-read.csv(dataURL,header = header)
  # which one is the class attribute
  # Class<-d[,as.integer(args[3])]
  classPosition<-as.integer(args[3])
  #classPosition<-6
  #Class<-d[,as.integer(classPosition)]
  Class <- as.factor(as.integer(classPosition))
  
  noOfCols<-ncol(d)
  
  for (j in 1:noOfCols){
    colName <- paste("V",j,sep="")
    colnames(d)[j]<-colName
  }
  
  classValue <- paste("V", Class,sep="")
  formula <- as.formula(paste(classValue," ~ .",sep=""))
  
  d[,classPosition]<-as.factor(d[,classPosition])
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  
  
  # now create all the classifiers and output accuracy values:
  
  #Decision Tree
  dec_fit<-rpart(formula,parms=list(split='information'), minsplit=2, minbucket=1,data=trainingData,method='class')
  #str(trainingData)
  #summary(dec_fit)
  par(mar = rep(0.4, 4))
  plot(dec_fit)
  text(dec_fit)
  #printcp(dec_fit)
  CPValue<-dec_fit$cptable[which.min(dec_fit$cptable[,"xerror"]),"CP"]
  pruned_fit<-prune(dec_fit,cp=as.numeric(CPValue))
  out_DecTree<-predict(pruned_fit,newdata=testData,type="vector")
  table_DecTree <- table(out_DecTree, testData[,classPosition])
  accuracy_decTree<-sum(diag(table_DecTree))/sum(table_DecTree)
  cat("Accuracy for Decision Tree for sample: ",i," is : ",accuracy_decTree*100,"% \n")
  decTree_accuracySum<-decTree_accuracySum+accuracy_decTree
  
  
  
  
  #SVM Default model
    svmModel <- svm(formula, data=trainingData)
  
  #summary(svmModel)
  out_SVM<-predict(svmModel,newdata=testData)
  table_SVM<-table(out_SVM, testData[,classPosition])
  accuracy_SVM<-sum(diag(table_SVM))/sum(table_SVM)
  svm_accuracySum<-svm_accuracySum+accuracy_SVM
  #accuracyPer_SVM<-accuracy_SVM*100
  cat("The accuracy for default SVM for sample ",i," is: ",accuracy_SVM*100,"% \n")
  
  
  #Naive Bayes Classifier
  
  naiveModel <- naiveBayes(formula, data = trainingData)
  out_naive<-predict(naiveModel,newdata=testData)
  table_naive<-table(out_naive, testData[,classPosition])
  accuracy_naive<-sum(diag(table_naive))/sum(table_naive)
  naive_accuracySum<-naive_accuracySum+accuracy_naive
  #accuracyPer<-accuracy*100
  cat("The accuracy for naive Bayes for sample ",i," is: ",accuracy_naive*100,"% \n")
  
  
  #kNN Classifier
  k <- 3
  Classvar <- trainingData[[classValue]]
  actual <- testData[[classValue]]
  trainingData[[classValue]] <- NULL
  testData[[classValue]] <- NULL          
  trainingData[,-1] <- as.double(gsub("%", "", as.matrix(trainingData[,-1])))
  testData[,-1] <- as.double(gsub("%", "", as.matrix(testData[,-1])))	
  trainingData[is.na(trainingData)] <- 0.0  
  testData[is.na(testData)] <- 0.0  	             
  predicted <- knn(trainingData, testData, Classvar, k, l = 0, prob = FALSE, use.all = TRUE)
  count=0
  arr <- 1:length(actual)
  for(value in arr){
    if(actual[value]==predicted[value]){
      count <- count+1
    }
  }
  accuracy_knn <-count/length(actual)
  knn_accuracySum<-knn_accuracySum+accuracy_knn
  cat("The accuracy for knn classifier for sample ",i," is: ",accuracy_knn*100,"% \n")
  
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  
  #Logistic Regression
  head(d)
  #summary(mydata)
  sapply(d, sd)
  mylogit <- glm(formula, data = trainingData, family = "binomial")
  
  #summary(mylogit)
  confint(mylogit)
  # How to predict
  # create test data
  #newdata <- with(testData, data.frame(gre = gre, gpa = gpa, rank = rank))
  # use predict with type="response", it will give you a probability of 1
  p<-predict(mylogit, newdata=testData, type="response")
  # use a threshold value and anything above that, you can assign to class=1 others to class=0
  threshold=0.50
  prediction<-sapply(p, FUN=function(x) if (x>threshold) 1 else 0)
  actual<-testData[,classPosition]
  actualIntArr<-as.integer(actual)
  for(index in 1:length(actualIntArr)){
    if(actualIntArr[index]==1)
    {
      actualIntArr[index]<-0
    }
    
  }
  for(index in 1:length(actualIntArr)){
    if(actualIntArr[index]==2)
    {
      actualIntArr[index]<-1
    }
    
  }
  
  accuracy_lr <- sum(actualIntArr==prediction)/length(actual)
  lr_accuracySum<-lr_accuracySum+accuracy_lr
  #print(accuracy)
  cat("The accuracy for Logistic Regression classifier for sample ",i," is: ",accuracy_lr*100,"% \n")
  
  #Bagging
  
  baggingModel <- bagging(formula, data=trainingData, mfinal=10)
  out_bagging <- predict(baggingModel,testData)	
  bagging_accuracy<-(sum(diag(out_bagging$confusion))/sum(out_bagging$confusion))
  bagging_accuracySum<-bagging_accuracySum+bagging_accuracy
  cat("The accuracy for Bagging classifier for sample ",i," is: ",bagging_accuracy*100,"% \n")
  
  #Random Forest
  
  randomForestModel <- randomForest(formula, data=trainingData)
  out_rf <- predict(randomForestModel,testData)
  accuracy_rf <-(sum(diag(table(out_rf, testData[[classValue]])))/sum(table(out_rf, testData[[classValue]])))
  randomForest_accuracySum<-randomForest_accuracySum+accuracy_rf
  #print(accuracy)
  cat("The accuracy for Random Forest classifier for sample ",i," is: ",accuracy_rf*100,"% \n")
  
  
  #Boosting
  
  boostingModel <- boosting(formula,data=trainingData,mfinal=20, coeflearn="Freund",boos=FALSE , control=rpart.control(maxdepth=3))
  out_boosting <- predict(boostingModel,testData)	
  boosting_accuracy<-(sum(diag(out_boosting$confusion))/sum(out_boosting$confusion))
  boosting_accuracySum<-boosting_accuracySum+boosting_accuracy
  cat("The accuracy for Boosting classifier for sample ",i," is: ",boosting_accuracy*100,"% \n")
  
  
  tryCatch({
  #Neural Network
  trainingData1<-d[sampleInstances,]
  testData1<-d[-sampleInstances,]
  n<-names(d)
  trainingData1[,as.integer(classPosition)] <- as.numeric(trainingData1[,as.integer(classPosition)])
  trainingData1[] <- lapply(trainingData1, function(x) as.numeric(as.character(x),na.rm=TRUE))
  Class <- trainingData1[,as.integer(classPosition)]
  f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))
  nnmodel <- neuralnet(f, trainingData1, hidden = 8, lifesign = "none", linear.output = FALSE, threshold = 0.1)
  plot(nnmodel, rep="best")
  
  testData1[,classPosition]<-as.integer(testData1[,classPosition])
  result <- compute(nnmodel, testData1)
  predicted <- round(result$net.result)
  nn_table <- table(predicted, as.factor(testData1[,as.integer(classPosition)]))
  
  accuracy_nn<-sum(diag(nn_table))/sum(nn_table)
  nn_accuracySum<-nn_accuracySum+accuracy_nn
  cat("The accuracy for Neural Network classifier for sample ",i," is: ",accuracy_nn*100,"% \n")
  })
  
  
  if(i==1){
    Sample1 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==2){
    Sample2 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==3){
    Sample3 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==4){
    Sample4 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==5){
    Sample5 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==6){
    Sample6 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==7){
    Sample7 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==8){
    Sample8 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==9){
    Sample9 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  else if(i==10){
    Sample10 = c(accuracy_decTree*100,accuracy_SVM*100,accuracy_naive*100,accuracy_knn*100,accuracy_lr*100,bagging_accuracy*100,accuracy_rf*100,boosting_accuracy*100)
  }
  # example of how to output
  # method="kNN" 
  # accuracy=0.9
  # cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
}


method<-"Decision Tree"
decTree_avgAccuracy<-decTree_accuracySum/10
cat("Method = ", method,", accuracy= ", decTree_avgAccuracy*100," % \n")

method<-"SVM Default"
svm_avgAccuracy<-svm_accuracySum/10
cat("Method = ", method,", accuracy= ", svm_avgAccuracy*100," % \n")

method<-"Naive Bayes"
naive_avgAccuracy<-naive_accuracySum/10
cat("Method = ", method,", accuracy= ", naive_avgAccuracy*100," % \n")

method<-"kNN"
knn_avgAccuracy<-knn_accuracySum/10
cat("Method = ", method,", accuracy= ", knn_avgAccuracy*100," % \n")

method<-"Logistic Regression"
lr_avgAccuracy<-lr_accuracySum/10
cat("Method = ", method,", accuracy= ", lr_avgAccuracy*100," % \n")

method<-"Bagging"
bagging_avgAccuracy<-bagging_accuracySum/10
cat("Method = ", method,", accuracy= ", bagging_avgAccuracy*100," % \n")

method<-"Random Forest"
rf_avgAccuracy<-randomForest_accuracySum/10
cat("Method = ", method,", accuracy= ", rf_avgAccuracy*100," % \n")

method<-"Boosting"
boostin_avgAccuracy<-boosting_accuracySum/10
cat("Method = ", method,", accuracy= ", boostin_avgAccuracy*100," % \n")

method<-"Neural Network"
neural_avgAccuracy<-nn_accuracySum/10
cat("Method = ", method,", accuracy= ", neural_avgAccuracy*100," % \n")

Method = c('Decision Tree','Support Vector Machines','Naive Bayesian','kNN','Logistic Regression','Bagging','Random Forest','Boosting')
Average = c(decTree_avgAccuracy*100,svm_avgAccuracy*100,naive_avgAccuracy*100,knn_avgAccuracy*100,lr_avgAccuracy*100,bagging_avgAccuracy*100,rf_avgAccuracy*100,boostin_avgAccuracy*100)

BestParameters<-c('Decision Tree','Radial Kernel','Naive Bayes','k=11','threshold=0.5','mfinal=10','Random Forest','maxdepth=3')
dataFrame4 = data.frame(Method,BestParameters,Sample1,Sample2,Sample3,Sample4,Sample5,Sample6,Sample7,Sample8,Sample9,Sample10,Average)
#dataFrame = data.frame(Method,BestParameters,Sample1,Sample2,Average)
