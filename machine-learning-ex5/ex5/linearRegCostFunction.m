function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X => (m x n)
% theta => (n x 1)
% h_theta => X * theta => (m x n) * (n x 1) => (m x 1)
h_theta = X * theta;
reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
% y => (m x 1)
% h_theta => (m x 1)
% h_theta - y => (m x 1) - (m x 1) = (m x 1)
J = (1/(2*m)) * sum((h_theta - y).^2) + reg_term;

% h_theta => (m x 1)
% y => (m x 1)
% X => (m x n)
% X' => (n x m)
% h_theta - y => (m x 1) - (m x 1) => (m x 1)
% (h_theta - y)' => (1 x m)
% X' * (h_theta - y) => (n x m) * (m x 1) => (n x 1)
grad = (1/m) * (X' * (h_theta - y));
grad(2:end) += (lambda/m) * theta(2:end);









% =========================================================================

grad = grad(:);

end
