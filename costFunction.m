function [J, grad] = costFunction(beta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(beta, X, y) computes the cost of using beta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples


predX = sigmoid(X * beta);
sqErr = (-y .* log(predX)) - ((1-y) .* log(1 - predX));
J = 1/m * (sum(sqErr));

err = predX - y;
w_err = sum(err' * X,1);
grad = 1/m .* w_err;


end
