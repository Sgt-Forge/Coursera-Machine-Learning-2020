function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X = m x n
% m = Number training examples
% n = Number of features 
% theta = n x 1
% n = number of features 
% h_theta = (m x n) * (n x 1) => (m x 1)
% Will return vector of predicted values
h_theta = X * theta;
% costs = (m x 1) - (m x 1) => (m x 1)
% subtraction element-wise
costs = h_theta - y;
% Calc the total cost for each training example
% Apply eleent-wise square to costs and sum the costs
J = (1/(2*m)) * (sum(costs.^2));


% =========================================================================

end
