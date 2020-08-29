function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% normal equation
% (X' * X)^(-1) * X' * y
% X => (m x n)
% X' => (n x m)
% X' * X = (n x m) * (m x n) => (n x n)
% inverse( X' * X ) => inverse of (n x n) => (n x n)
% inv * X' => (n x n) * (n x m) => (n x m)
% y = (m x 1)
% (inv * X') * y => (n x m) * (m x 1) => n x 1
X_prime_X = X' * X;
X_prime_inv = pinv(X_prime_X);
X_inv_t = X_prime_inv * X';
theta = X_inv_t * y;
% One line
% theta = pinv(X' * X) * X' * y;

% ---------------------- Sample Solution ----------------------




% -------------------------------------------------------------


% ============================================================

end
