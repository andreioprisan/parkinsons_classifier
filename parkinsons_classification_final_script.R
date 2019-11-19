library(readr)
train <- read_csv("Downloads/2019-sa-101c/train.csv")
test <- read_csv("Downloads/2019-sa-101c/test.csv")

#change variables from (2,1) to (1,0) for predicting
train$category01 <- train$category - 1




library(dplyr)
library(reshape2)
library(knitr)
corrMatrix <- cor(train)
corrWithCateg <- corrMatrix[dim(corrMatrix)[1],] 
corrVecCateg <- rev(corrWithCateg[order(abs(corrWithCateg))]) 
#df of variables w/ strongest corr w/ category
highestCorrCategDF <- data.frame(name = names(corrVecCateg), corr = corrVecCateg, stringsAsFactors = F) 

#selection of "strong predictors": any variable with a corr coeff of 0.2 or lower (wrt category) is discarded (the lowest qualifying variable was also discarded bc of interpretability reasons)
corrVecCateg <- corrVecCateg[abs(corrVecCateg) >= 0.2]
highestCorrCategDF <- highestCorrCategDF[(abs(highestCorrCategDF$corr) >= 0.2),]
highestCorrCategDF <- highestCorrCategDF[c(-1, -2), ]
rownames(highestCorrCategDF) <- c()
highestCorrCategDF %>% head(10L) %>% kable




library(stringr)
getPCA <- function(train, test)
{
  trainData <- train %>% select(`Log energy`, category01)
  testData <- test %>% select(`Log energy`)
  
  var_DYS1 <- highestCorrCategDF[str_detect(highestCorrCategDF$name, "Jitter|Shimmer"), ]$name
  var_DYS2 <- highestCorrCategDF[str_detect(highestCorrCategDF$name, "PPE|RPDE|GQ|HNR|DFA|GNE|VFER|SNR"), ] %>% filter(!name %in% c("IMF->SNR_TKEO", "IMF->SNR_entropy")) %>% select(name) %>% unlist
  var_MFCC <- highestCorrCategDF[str_detect(highestCorrCategDF$name, "MFCC"), ]$name
  var_IMF <- highestCorrCategDF[str_detect(highestCorrCategDF$name, "IMF"), ]$name 
  
  DYS1 <- train[, which(names(train) %in% var_DYS1)] 
  DYS2 <- train[, which(names(train) %in% var_DYS2)]
  MFCC <-  train[, which(names(train) %in% var_MFCC)]
  IMF <-  train[, which(names(train) %in% var_IMF)]
  
  DYS1_PCA <- prcomp(DYS1, center = T, scale = T)
  DYS2_PCA <- prcomp(DYS2, center = T, scale = T)
  MFCC_PCA <- prcomp(MFCC, center = T, scale = T)
  IMF_PCA <- prcomp(IMF, center = T, scale = T)
  
  trainData$DYS1_PC1 <- DYS1_PCA[["x"]][,1]
  trainData$DYS2_PC1 <- DYS2_PCA[["x"]][,1]
  trainData$MFCC_PC1 <- MFCC_PCA[["x"]][,1]
  trainData$IMF_PC1 <- IMF_PCA[["x"]][,1]
  
  trainData$MFCC_PC2 <- MFCC_PCA[["x"]][,2]
  trainData$DYS1_PC2 <- DYS1_PCA[["x"]][,2]
  trainData$DYS2_PC2 <- DYS2_PCA[["x"]][,2]
  
  trainData$DYS1_PC3 <- DYS1_PCA[["x"]][,3]
  trainData$DYS2_PC3 <- DYS2_PCA[["x"]][,3]
  
  trainData$DYS1_PC4 <- DYS1_PCA[["x"]][,4]
  trainData$DYS1_PC5 <- DYS1_PCA[["x"]][,5]
  
  testData$DYS1_PC1 <- scale(test[, which(names(test) %in% var_DYS1)], center = DYS1_PCA$center) %*% DYS1_PCA$rotation[,1] %>% c 
  testData$DYS2_PC1 <- scale(test[, which(names(test) %in% var_DYS2)], center = DYS2_PCA$center) %*% DYS2_PCA$rotation[,1] %>% c
  testData$MFCC_PC1 <- scale(test[, which(names(test) %in% var_MFCC)], center = MFCC_PCA$center) %*% MFCC_PCA$rotation[,1] %>% c
  testData$IMF_PC1 <- scale(test[, which(names(test) %in% var_IMF)], center = IMF_PCA$center) %*% IMF_PCA$rotation[,1] %>% c
  
  testData$MFCC_PC2 <- scale(test[, which(names(test) %in% var_MFCC)], center = MFCC_PCA$center) %*% MFCC_PCA$rotation[,2] %>% c
  testData$DYS1_PC2 <- scale(test[, which(names(test) %in% var_DYS1)], center = DYS1_PCA$center) %*% DYS1_PCA$rotation[,2] %>% c
  testData$DYS2_PC2 <- scale(test[, which(names(test) %in% var_DYS2)], center = DYS2_PCA$center) %*% DYS2_PCA$rotation[,2] %>% c
  
  testData$DYS1_PC3 <- scale(test[, which(names(test) %in% var_DYS1)], center = DYS1_PCA$center) %*% DYS1_PCA$rotation[,3] %>% c
  testData$DYS2_PC3 <- scale(test[, which(names(test) %in% var_DYS2)], center = DYS2_PCA$center) %*% DYS2_PCA$rotation[,3] %>% c
  
  
  testData$DYS1_PC4 <- scale(test[, which(names(test) %in% var_DYS1)], center = DYS1_PCA$center) %*% DYS1_PCA$rotation[,4] %>% c
  
  testData$DYS1_PC5 <- scale(test[, which(names(test) %in% var_DYS1)], center = DYS1_PCA$center) %*% DYS1_PCA$rotation[,5] %>% c
  
  return(list(trainData, testData))
}




#NN REVISITED


library(neuralnet)
set.seed(1)
fan_in <- 12
startWeights <- rnorm(100000, mean = 0, sd = sqrt(2 / fan_in))

errorVecTr <- rep(NA, 12)
errorVecTe <- rep(NA, 12)
bestSize1 <- c()

for(j in 1:4)
{
  testIndexStart <- (j*22 - 21)
  testIndexEnd <- testIndexStart + 21
  testIndex <- testIndexStart:testIndexEnd
  
  subTrain <- train[-testIndex, ]
  subTest <- train[testIndex, ]
  
  trainLabels <- subTrain$category01
  testLabels <- subTest$category01
  
  a <- getPCA(subTrain, subTest)
  trainData <- a[[1]] 
  testData <- a[[2]]
  scaledTr <- trainData
  scaledTe <- testData
  names(scaledTr)[1] <- "Log_energy"
  names(scaledTe)[1] <- "Log_energy"
  
  for(i in 1:12)
  {
    parkModel1 <- neuralnet(category01 == 1 ~., hidden = i, data = scaledTr, linear.output= F, learningrate = 0.01, threshold = 2.5, algorithm = 'rprop+', startweights = startWeights, act.fct = 'tanh')
    
    subTrainResults <- round(predict(parkModel1, scaledTr)) %>% as.vector
    subTestResults <- round(predict(parkModel1, scaledTe)) %>% as.vector
    
    errorVecTr[i] <- sum(abs(trainLabels - subTrainResults) ) / 66
    errorVecTe[i] <- sum(abs(testLabels- subTestResults) ) / 22
  }
  errDF1 <- data.frame(TrainError = errorVecTr, TestError = errorVecTe, I = 1:12, J = j)
  bestSize1 <- rbind(bestSize1, errDF1)
}

summaryError1 <- bestSize1 %>% group_by(I) %>% summarise(meanTest = mean(TestError), meanTrain = mean(TrainError), medianTest = median(TestError), medianTrain = median(TrainError), testSD = sd(TestError))






set.seed(1)
fan_in <- 12
startWeights <- rnorm(56, mean = 0, sd = sqrt(2 / fan_in))

a <- getPCA(train, test)
trainData <- a[[1]]
testData <- a[[2]]

names(trainData)[1] <- "Log_energy"
names(testData)[1] <- "Log_energy"
parkModel <- neuralnet(category01 == 1 ~., hidden = 2, data = trainData, linear.output= F, learningrate = 0.01, threshold = 2.5, algorithm = 'rprop+', startweights = startWeights, act.fct = 'tanh')
testResults <- (round(predict(parkModel, testData)) + 1) %>%as.vector





submitPredictions <- function(submission)
{
  test_ids <- 89:126
  output <- data.frame(id = test_ids, category = submission)
  write.csv(output, 'FINAL_classification_upload.csv', quote = FALSE, row.names = FALSE)
}

submitPredictions(testResults)

