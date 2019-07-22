clc
close all
clear

%Predicting whether a match came from the WTA or the ATP tour from match stats
%This script learns a random forest (bagged tree) using the stats toolbox

data = readtable('tour_dataset.csv');

%sample training and test data
rng(2, 'twister');
[trainInd, ~, testInd] = dividerand(height(data), 0.75, 0, 0.25);
Y_train = data(trainInd, 1);
X_train = data(trainInd, 2:11);
Y_test = data(testInd, 1);
X_test = data(testInd, 2:11);

B = TreeBagger(500, X_train, Y_train, 'Method', 'classification');

Y_pred = predict(B, X_test);

%create confusion matrix
confMat = confusionmat(table2array(Y_test), str2double(Y_pred));

accuracy = sum(diag(confMat))/sum(sum(confMat)); %compute training prediction accuracy

%compute precision, recall and F-scores
precision1 = confMat(2,2)/(sum(confMat(:,2)));
recall1 = confMat(2,2)/(sum(confMat(2, :)));
fscore1 = 2*precision1*recall1/(precision1 + recall1);
precision0 = confMat(1,1)/(sum(confMat(:,1)));
recall0 = confMat(1,1)/(sum(confMat(1,:)));
fscore0 = 2*precision0*recall0/(precision0 + recall0);