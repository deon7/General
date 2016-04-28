##DEON CRASTO
##HW3

library(caret)
library(data.table)
library(stringr)
library(e1071)
library(cwhmisc)
library(reshape2)
library(ggplot2)
library(rpart)

install.packages('caret')
install.packages('stringr')
install.packages('minqa')

record_performance <- function(df, name, model, test) {
  svm.pred <- predict(model, test)
  svm.table <- table(pred = svm.pred, true=test$corr)
  df <- rbind(df, data.frame(SVM_Model=c(name), Accuracy=c(classAgreement(svm.table)$diag)))
  return(df)
}
decision_tree_performance <- function(table_name, var_name, model_name, data_name) {
  pred_tree <- predict(model_name, data_name)
  
  predictions <- data.table(cbind(data_name$corr,pred_tree))
  predictions[, predict := ifelse(False > True, 1, 2)]
  
  accuracy<-confusionMatrix(predictions$V1, predictions$predict)
  accr<-accuracy$overall['Accuracy']
  result_perc <- cbind(var_name,accr)
  table_name <- rbind(table_name, result_perc)
  return(table_name)
}

paren_match <- function(page, text) {
  start <- cpos(page, "(")
  end <- cpos(page, ")")
  if (!is.na(start) && !is.na(end)) {
    search <- substring(text, start + 1, end - 1)
    return(grepl(tolower(search), tolower(text), fixed=TRUE))
  } else {
    return(FALSE)
  }
}


#Loading File
full <- read.csv("C:\\Users\\Deon\\Downloads\\qb.train.csv")

#Finding number of characters in text
full$obs_len <- apply(full, 1, function(x) {nchar(x['text'])})
full$scale_len <- scale(full$obs_len)

full$scale_score <- scale(full$body_score)

#Finding if parenthesis present or not
full$paren_match <- apply(full, 1, function(x) {paren_match(x['page'], x['text'])})

#Taking log of in-links and scaling
full$log_links <- scale(log(as.numeric(full$inlinks) + 1))

#New feature
#Finding Total number of commas in text
full$commas_total <- str_count(full$text, ",")
full$log_commas <- scale(log(as.numeric(full$commas_total) + 1))
##Testing data
testing_data <- read.csv("qb.test.csv")
testing_data$obs_len <- apply(testing_data, 1, function(x) {nchar(x['text'])})

#Scale characters
testing_data$scale_len <- scale(testing_data$obs_len)

#Scale bodyscore
testing_data$scale_score <- scale(testing_data$body_score)

#Paranthesis check
testing_data$paren_match <- apply(testing_data, 1, function(x) {paren_match(x['page'], x['text'])})


testing_data$log_links <- scale(log(as.numeric(testing_data$inlinks) + 1))
testing_data$commas_total <- str_count(testing_data$text, ",")
testing_data$log_commas <- scale(log(as.numeric(testing_data$commas_total) + 1))
index <- 1:nrow(full)

#Sampling with 20% of total data 
testindex <- sample(index, trunc(length(index)/5))
testset <- full[testindex,]
trainset <- full[-testindex,]

#SVM
mfc <- sum(testset$corr == "False") / nrow(testset)

#different model evaluation
results <- data.frame(SVM_Model=c("MFC"), Accuracy=c(mfc))
results <- record_performance(results, "scale_score", svm(corr ~ scale_score, data=trainset), testset)
results <- record_performance(results, "scale_len", svm(corr ~ scale_len, data=trainset), testset)
results <- record_performance(results, "log_commas", svm(corr ~ log_commas, data=trainset), testset)
results <- record_performance(results, "paren_match", svm(corr ~ paren_match, data=trainset), testset)
results <- record_performance(results, "log_links", svm(corr ~ log_links, data=trainset), testset)
results <- record_performance(results, "log_commas + scale_score", svm(corr ~ log_commas + scale_score, data=trainset), testset)
results <- record_performance(results, "log_commas + log_links", svm(corr ~ log_commas + log_links, data=trainset), testset)
results <- record_performance(results, "scale_len + log_commas", svm(corr ~ scale_len + log_commas, data=trainset), testset)
results <- record_performance(results, "log_commas + scale_score + log_links", svm(corr ~ log_commas + scale_score + log_links, data=trainset), testset)
results <- record_performance(results, "log_commas + scale_score + paren_match", svm(corr ~ log_commas + scale_score + paren_match, data=trainset), testset)
results <- record_performance(results, "scale_len + scale_score + log_commas", svm(corr ~ scale_len + scale_score + log_commas, data=trainset), testset)
results <- record_performance(results, "scale_len + scale_score + log_links + paren_match + log_commas", svm(corr ~ scale_len + scale_score + log_links + paren_match + log_commas, data=trainset), testset)

results

#Decision Trees model

resultset <- data.frame("Decision_Tree_Model" = character(), "Accuracy" = numeric())
resultset <- decision_tree_performance(resultset, "scale_score", rpart(corr ~ scale_score, data=trainset, control = rpart.control(minsplit = 1)), testset)
resultset <- decision_tree_performance(resultset, "paren_match", rpart(corr ~ paren_match, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "log_links", rpart(corr ~ log_links, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "log_commas + scale_score", rpart(corr ~ log_commas + scale_score, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "log_commas + log_links", rpart(corr ~ log_commas + log_links, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "log_commas + paren_match", rpart(corr ~ log_commas + paren_match, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "log_commas + scale_score + log_links", rpart(corr ~ log_commas + scale_score + log_links, data=trainset, control = rpart.control(minsplit = 1)), testset)
resultset <- decision_tree_performance(resultset, "log_commas + scale_score + paren_match", rpart(corr ~ log_commas + scale_score + paren_match, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "scale_len + scale_score + log_commas", rpart(corr ~ scale_len + scale_score + log_commas, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset <- decision_tree_performance(resultset, "scale_len + scale_score + log_links + paren_match + log_commas", rpart(corr ~ scale_len + scale_score + log_links + paren_match + log_commas, data=trainset, control = rpart.control(minsplit = 20, minbucket = 6)), testset)
resultset



testset$corr_numeric <- as.numeric(testset$corr)
testset$corr_numeric <- ifelse(testset$corr_numeric == 1,0,1)
test_glm <- data.table(testset)
results_logistic <- data.frame("Logistic_Model" = character(), "Accuracy" = numeric())



#models using different paramters
glmodel1 <- glm(corr ~ as.vector(body_score), family = binomial(link = 'logit'), data = trainset)
f.results_glm_1 <- predict(glmodel1, test_glm, type = 'response')
f.results_glm_1 <- ifelse(f.results_glm_1 > 0.5,1,0)
misClassificError_glm_1 <- mean(f.results_glm_1 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model=c('body_score'), Accuracy=c(1-misClassificError_glm_1)))


glmodel2 <- glm(corr ~ as.vector(commas_total), family = binomial(link = 'logit'), data = trainset)
f.results_glm_2 <- predict(glmodel2, test_glm, type = 'response')
f.results_glm_2 <- ifelse(f.results_glm_2 > 0.5,1,0)
misClassificError_glm_2 <- mean(f.results_glm_2 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='commas_total', Accuracy = 1 - misClassificError_glm_2))

glmodel3 <- glm(corr ~ as.vector(inlinks), family = binomial(link = 'logit'), data = trainset)
f.results_glm3 <- predict(glmodel3, test_glm, type = 'response')
f.results_glm3 <- ifelse(f.results_glm3 > 0.5,1,0)
misClassificError_glm3 <- mean(f.results_glm3 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='inlinks', Accuracy = 1 - misClassificError_glm3))

glmodel4 <- glm(corr ~ paren_match, family = binomial(link = 'logit'), data = trainset)
f.results_glm4 <- predict(glmodel4, test_glm, type = 'response')
f.results_glm4 <- ifelse(f.results_glm4 > 0.5,1,0)
misClassificError_glm4 <- mean(f.results_glm4 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='paren_match', Accuracy = 1 - misClassificError_glm4))

glmodel5 <- glm(corr ~ as.vector(obs_len) + as.vector(commas_total), family = binomial(link = 'logit'), data = trainset)
f.results_glm5 <- predict(glmodel5, test_glm, type = 'response')
f.results_glm5 <- ifelse(f.results_glm5 > 0.5,1,0)
misClassificError_glm5 <- mean(f.results_glm5 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='scale_len + commas_total', Accuracy = 1 - misClassificError_glm5))

glmodel6 <- glm(corr ~ as.vector(obs_len) + as.vector(inlinks), family = binomial(link = 'logit'), data = trainset)
f.results_glm6 <- predict(glmodel6, test_glm, type = 'response')
f.results_glm6 <- ifelse(f.results_glm6 > 0.5,1,0)
misClassificError_glm6 <- mean(f.results_glm6 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='obs_len + upper_case', Accuracy = 1 - misClassificError_glm6))

glmodel7 <- glm(corr ~ as.vector(obs_len) + as.vector(body_score) + as.vector(inlinks), family = binomial(link = 'logit'), data = trainset)
f.results_glm7 <- predict(glmodel7, test_glm, type = 'response')
f.results_glm7 <- ifelse(f.results_glm7 > 0.5,1,0)
misClassificError_glm7 <- mean(f.results_glm7 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='obs_len + body_score + inlinks', Accuracy = 1 - misClassificError_glm7))

glmodel8 <- glm(corr ~ as.vector(obs_len) + as.vector(body_score) + as.vector(commas_total), family = binomial(link = 'logit'), data = trainset)
f.results_glmodel8 <- predict(glmodel8, test_glm, type = 'response')
f.results_glmodel8 <- ifelse(f.results_glmodel8 > 0.5,1,0)
misClassificError_glmodel8 <- mean(f.results_glmodel8 != test_glm$corr_numeric)
results_logistic <- rbind(results_logistic, data.frame(Logistic_Model ='obs_len + body_score + commas_total', Accuracy = 1 - misClassificError_glmodel8))

results_logistic


best_fit_model <- svm(corr ~ scale_len + scale_score + log_links + paren_match + log_commas, data=full)
ans <-  predict(best_fit_model, testing_data)
submit <- data.frame(row = testing_data$row, corr = ans)
write.csv(submit, file = "final.csv", row.names = FALSE)

#Plotting

ggplot(full, aes(commas_total, fill = corr)) + geom_histogram() + facet_grid(~ corr) + ggtitle('Number of Commas in Text')


