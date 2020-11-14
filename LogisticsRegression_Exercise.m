%% Exercise: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that covers regularization with logistic 
%  regression. 
%
%   
%   To run the code, you can copy each chunk and paste it in the Matlab
%   terminal, or highlight the specific lines and evaluate code.
%
%
%   Code used in this exercise are edited from Coursera's machine learning
%   course by Andrew Ng (~80%), and course assignments from NUS.
%
%   Edited by JH POH - 2/16/2020
%
%   ********* Important Notes **********
%   Please note that many of the code and function implemented here have
%   been simplified to make things readable and understandable. You may
%   notice that there are several hard-coded parameters and these
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

%% ------------------- Part 1 (Logistics regression) ---------------------
%% Load Data 1
% This is a set of data that is linearly separable. Each participant has 2
% scores, and the goal is to predict whether they belong to Class A or
% Class B using the 2 scores.
% The first two columns contains the X values and the third column
% contains the binary label (y).

data = load('data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

% Given that we are only looking at 2 dimensions (features), we can
% visualise the data.

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Voxel 1')
ylabel('Voxel 2')

% Specified in plot order
legend('ClassA', 'ClassB')
hold off;

%% Starting the regression
% *** NOTES ***
% We will next perform a logistics regression to obtain our parameter
% estimate (beta). While you can easily implement logistics regression with
% a single function call in most packages (e.g. in matlab you can use
% glmfit, fitlm, fitclinear etc etc), but over here, to get a sense of how
% the cost function changes at different beta, we will be using fminunc - a
% function that minimize the cost function of your choice using gradient
% descent. The cost function used here is implemented in costFunction.m
% and fits the data to a sigmoid at each beta, and computes the squared
% error term.
% For Python users, this can be implemented with scikit-learn.


% Setting up the data matrix and add a column of ones for the intercept
% term.
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term
X = [ones(m, 1) X];

% Initialize fitting parameters (Start with 0 for all parameters)
% (You can change this parameter to see how the value change at different
% cost function)
initial_beta = zeros(n + 1, 1);  


% Compute and display initial cost and gradient
[cost, ~] = costFunction(initial_beta, X, y);

fprintf('Cost at initial beta: %f\n', cost);


% Find optimal beta using fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal beta
%  This function will return beta and the cost
[beta, cost] = ...
	fminunc(@(b)(costFunction(b, X, y)), initial_beta, options);

% Print parameter estimates to screen --
% * Note we have 2 features and an intercept, so there should be 3 values
%   corresponding to each of the terms.
fprintf('Cost at beta found by fminunc: %f\n', cost);
fprintf('beta: \n');
fprintf(' %f \n', beta);


% Plot Boundary
plotDecisionBoundary(beta, X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Voxel 1')
ylabel('Voxel 2')
% Specified in plot order
legend('ClassA', 'ClassB')
hold off;


%%  Generating predictions!
% Now that we have created a classifier, we can compute the accuracy of our
% training data set by fitting the data to our model!
p = predict(beta,X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

% And since we now have a model, we can also generate predictions! To
% generate a prediction, you can input the value of your 2 features, and it
% will produce a probablity of which class this item would likely belong
% to!
% 
% In actual fMRI analysis, this would typically be our Test data set. E.g.
% If we train our model based on a portion of the data, can this model 
% predict the items given Activation pattern from the left out data?
% 
% Try changing the input value for voxel 1 and voxel 2, to see the
% prediction it generates. 
vx1 = 10;
vx2 = 100;

prob = sigmoid([1 vx1 vx2] * beta);
fprintf(['For activation of %f in Voxel 1 and %f in Voxel 2, ' ...
         'we predict Class A with a ' ...
         'probability of %f\n\n'], vx1,vx2, prob);
     
% ****** EXERCISE TIME! ********
% Try splitting the data above into a training and test set. For a start,
% try using 80% of the data for training, and then apply your model on the
% test set (For simplicity, I have created 2 separate matrix for classA and
% class B items).
% How well does it work?
% 
% ** Note: Keep the dataset as balanced as possible -- i.e. similar number 
% of training examples for Class A and Class B. 
%
% Why do you think this is important?

% classA = data((data(:,3)==1),:);
% classB = data((data(:,3)==0),:);
% ------------ Your code here -----------
% 
% Randomly select a subset from each dataset
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

% Training the model with our new training set
[m, n] = size(trainX);
initial_beta = zeros(n, 1);  

[beta, cost] = ...
	fminunc(@(b)(costFunction(b, trainX, trainy)), initial_beta, options);
p = predict(beta,trainX);
fprintf('Training Accuracy: %f\n', mean(double(p == trainy)) * 100);

% Plot Training output
plotDecisionBoundary(beta, trainX, trainy);
% Put some labels 
hold on;
xlabel('Voxel 1')
ylabel('Voxel 2')
legend('ClassA', 'ClassB')
title('Training classification')
hold off;


% Testing on the held out data with our new model
[m, n] = size(testX);

p = predict(beta,testX);
fprintf('Test Accuracy: %f\n', mean(double(p == testy)) * 100);

% Plot Test Output
plotDecisionBoundary(beta, testX, testy);
% Put some labels 
hold on;
xlabel('Voxel 1')
ylabel('Voxel 2')
legend('ClassA', 'ClassB')
title('Test classification')
hold off;

     
%% ------------------- Part 2 (Regularization) ---------------------
%% Load Data
%  In this part, we will look at a data set that is not linearly separable
%  when looking only at 2 features. One way of getting around this, is by
%  adding more features, in the form of polynomial terms. 
%
%  Note: While the addition of polynomial term is extremely common in ML,
%  it is typically not used in fMRI analysis, given that we typically have
%  wayyyyyyyy more feature than data. Over here, the main goal is to look
%  at how Regularization can influence our Decision boundary.
%  
%  Similary, the first two columns contains the X values and the third 
%  column contains the label (y).
clc; clear;

data = load('data2.txt');
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

% Add Polynomial Features - over here we are adding more features, creating
% X1, X1^2, X1*X2 X1^2*X2^2 etc etc etc. So while we started with 2
% features, we now have 28 features (you can try increasing or decreasing
% it further with the parameter 'degree'). 

degree = 6;
out = ones(size(X(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X(:,1).^(i-j)).*(X(:,2).^j);
    end
end
X = out;

% Initialize fitting parameters
initial_beta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_beta, X, y, lambda);

fprintf('Cost at initial beta (zeros): %f\n', cost);


% Similar to the above, we now try to find our parameter estimates and to
% obtain our decision boundary. But this time, you would notice that there
% is an additional parameter "lambda". 
% 
% Lambda here is a regularization parameter. Over here, we are using
% L2-regularisation (look at the script costFunctionReg.m if you are
% interested in the details). Regularization penalises high parameter
% estimates, and prevents over fitting. This is especially important when
% your feature space is large, and your data set is limited.
% 
% Think of the parameter as adjusting how 'flexible' your decision boundary
% can be.
% 
% ****** EXERCISE TIME! ********
% In the code below, try changing the lambda parameter and report your
% Training accuracy across a range of Lambda. 
% How does the accuracy change? 
%
% Look at your plot for different lambda. How does it vary?
% ------------ Change Here! -----------
lambda = 100;
% -------------------------------------


initial_beta = zeros(size(X, 2), 1);
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[beta, J, exit_flag] = ...
	fminunc(@(b)(costFunctionReg(b, X, y, lambda)), initial_beta, options);

% Plot Boundary
plotDecisionBoundary(beta, X, y,degree);
hold on;
title(sprintf('lambda = %g', lambda))
% Labels and Legend
xlabel('Voxel 1')
ylabel('Voxel 2')
% Specified in plot order
legend('ClassA', 'ClassB','Decision Boundary')
hold off;

% Compute accuracy on our training set
p = predict(beta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

% ****** EXERCISE TIME! ********
% Similar to the previous exercise, try splitting your dataset into
% a training and a test set.
%
% Train your model with different lambda, does it affect the performance on
% the training and test set differently?
% ------------ Your code here! -------------
%

lambda = 100;
test_pct = .2;

cv = cvpartition(y,'Holdout',test_pct);
trainX = X(cv.training,:);
trainy = y(cv.training,:);
testX = X(cv.test,:);
testy = y(cv.test,:);

% Training the model with our new training set
[m, n] = size(trainX);
initial_beta = zeros(n, 1);  

options = optimset('GradObj', 'on', 'MaxIter', 400);
[beta, J, exit_flag] = ...
	fminunc(@(b)(costFunctionReg(b, trainX, trainy, lambda)), initial_beta, options);

p = predict(beta,trainX);
fprintf('Training Accuracy: %f\n', mean(double(p == trainy)) * 100);

% Plot Training output
plotDecisionBoundary(beta, trainX,trainy, degree);
% Put some labels 
hold on;
xlabel('Voxel 1')
ylabel('Voxel 2')
legend('ClassA', 'ClassB')
title(sprintf('Training - lambda = %g', lambda))
hold off;


% Testing on the held out data with our new model
[m, n] = size(testX);
p = predict(beta,testX);
fprintf('Test Accuracy: %f\n', mean(double(p == testy)) * 100);

% Plot Test Output
plotDecisionBoundary(beta, testX, testy, degree);
% Put some labels 
hold on;
xlabel('Voxel 1')
ylabel('Voxel 2')
legend('ClassA', 'ClassB')
title(sprintf('Testing - lambda = %g', lambda))
hold off;
%
%
%



