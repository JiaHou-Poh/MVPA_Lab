%% Exercise: SVM
%
%  Instructions
%  ------------
% 
%  This file contains code that covers SVM.
%
%
%   To run the code, you can copy each chunk and paste it in the Matlab
%   terminal, or highlight the specific lines and evaluate code.
%
%
%   Code used in this exercise are edited from Coursera's Machine learning
%   course by Andrew Ng (~80%), and course assignments from NUS.
%
%   Edited by JH POH - 2/16/2020
%
%   ********* Important Notes **********
%   Please note that many of the code and function implemented here have
%   been simplified to make things readable and understandable. You may
%   notice that there are a good number of hard-coded parameters and these
%   are mostly done for simplicity (and kinda for my own convenience).
%   Some of the code are directly taken from different classes I have taken
%   over the years and have not been extensively tested, so DO NOT
%   use these for your own actual data analysis!!!
%
%   Functions described here are all common functions that have been built
%   into various well-established packages. Do use those instead! Once you
%   understand the parameters required for each of these functions, it's
%   super easy to just call those functions regardless of package.
%
%
%% Initialization
clear ; close all; clc

%% ------------------- Part 1 (Linear SVM) ---------------------
% For the first part, we will use the same dataset that we used in the
% logistics regression exercise, but instead of using logistics regression,
% we will now use a support vector machine (SVM).

% Similarly we start with some basic visualisations
data = load('data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Voxel 1')
ylabel('Voxel 2')
% Specified in plot order
legend('ClassA', 'ClassB')
hold off;

%% Starting the SVM
% *** NOTES ***
% Similar to the logistics regression example, there are numerous packages
% out there with different implementations of SVM and you do not have to
% write your own code. The svmTrain function here is adapted from 
% Coursera ML assignment and is made to be readable, but it's also
% inflexible and inefficient.
% Matlab has it's own implementation of
% SVM and can be called using fitclinear or fitcsvm. 
% 
% For Python users, there's scikit-learn. Libsvm has also recently added
% support for Python.

fprintf('\nTraining Linear SVM ...\n')

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure;visualizeBoundaryLinear(X, y, model);

% ****** EXERCISE TIME! ********
% Try changing the parameter "C" to different value and look at how the
% decision boundary changes.
% What do you think C is doing? 
%
%
%
%
%
%
%% ------------------- Part 2 (Non-Linear SVM) ---------------------
% The use of non-linear classifiers on fMRI data has been a longstanding
% debate. In general, non-linear classifiers are hardly ever used for fMRI
% analysis because there is simply not enough data (relative to the number
% of feature/dimensions).In fact, most of the time, we have to use feature
% reduction/ selection techniques to increase classification performance.
%
% For learning purposes,it might be useful to examine how
% non-linear SVM works, and how different values of C influence
% classification performance.
%
% Non-linear SVM is typically implemented using a method called the Kernel
% trick. In this example, we will be using a Gaussian Kernel (which is a
% type of RBF kernel).

% Load new data 
clear; close all; clc;

load('data3.mat');

plotData(X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Voxel 1')
ylabel('Voxel 2')
% Specified in plot order
legend('ClassA', 'ClassB')
hold off;


% Specify SVM Parameters
C = 0.001; sigma = 0.05;

fprintf('\nTraining SVM with Gaussian Kernel ...\n');
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

% ****** EXERCISE TIME! ********
% You may have noticed that for the non-linear SVM, we now have an
% additional free parameter, 'sigma'. Try different values of C and sigma
% to see how the decision boundary changes.
% * Note: Due to my shoddy coding, I think some combinations of C and sigma
% might not work. Not sure where the bug is, and I am just gonna leave it
% be since this is just a visualization exercise. Once again, DONT use
% this set of code for your actual analysis!!!
% 
% Now try splitting the data into a training and test set and see how
% different values of C changes your Training and Test accuracy.
%
% ------------ Your code here -----------
%
test_pct = .2; % percentage of trials to select for training

% Create your Training and Test sets. 
% Over here I am using a built-in function called cvpartition that can
% randomly split your data based on the proportion you select, but you can
% also manually select or use different permutations to do this splitting.
cv = cvpartition(y,'Holdout',test_pct);
trainX = X(cv.training,:);
trainy = y(cv.training,:);
testX = X(cv.test,:);
testy = y(cv.test,:);

C = 0.001; sigma = 0.05;

fprintf('\nTraining SVM with Gaussian Kernel ...\n');
model= svmTrain(trainX, trainy, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(trainX, trainy, model);
hold on;
legend('ClassA', 'ClassB')
title(sprintf('Training - C = %g, Sigma = %g', C, sigma))
hold off;

visualizeBoundary(testX, testy, model);
hold on;
legend('ClassA', 'ClassB')
title(sprintf('Test - C = %g, Sigma = %g', C, sigma))
hold off;
