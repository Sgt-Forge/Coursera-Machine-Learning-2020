function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% X => (m x n)
% theta => (n x 1)
% X * theta => (m x n) * (n x 1) => (m x 1)
h_theta = sigmoid(X * theta);
% sigmoid(X * theta) => sigmoid( (m x 1) ) => (m x 1) => h_theta
% y => (m x 1)
% h_theta => m x 1
% y' => (1 x m)
% y' * h_theta => (1 x m) * (m x 1) => (1 x 1)

% theta => (n-1 x 1)
reg = (lambda/(2*m)) * sum(theta(2:end).^2);
J = (1/m) * sum(-y' * log(h_theta) - (1 .- y)' * log(1 .- h_theta)) + reg;


% theta(2:end) => (n - 1 x 1)
% [0, theta(2:end)] => (1 + n-1 x 1)
reg2 = [0; (lambda/m) * theta(2:end)];

% h_theta => m x 1
% y => (m x 1)
% X => (m x n)
% h_theta - y => (m x 1)
% X' => (n x m)
% X' * h_theta => (n x m) * (m x 1) => (n x 1)
grad = (1/m) * (X' * (h_theta - y)) + reg2;



% =============================================================

end
