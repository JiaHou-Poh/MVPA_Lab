function [J, grad] = costFunctionReg(beta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(beta, X, y, lambda) computes the cost of using
%   beta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


t_num = length(beta);

predX = sigmoid(X * beta);
reg = (lambda/(2*m)) * (sum(beta(2:t_num).^2));
regErr = (-y .* log(predX)) - ((1-y) .* log(1 - predX));
J = 1/m * sum(regErr) + reg;


err = predX - y;
w_err = sum(err' * X,1);
temp_grad1 = 1/m .* w_err(1);
temp_grad = 1/m .* w_err(2:t_num) + (lambda/m).*beta(2:t_num)';

grad=[temp_grad1 temp_grad];


end
