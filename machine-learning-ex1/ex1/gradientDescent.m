function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % X => (m x n)
    % theta => (n x 1)
    % y => (m x 1)
    % error => (m x 1) - (m x 1)
    error = (X * theta) - y;
    % error => (m x 1)
    % Need to update thetas so we need (n x 1) - (n x 1)
    % X' => (n x m) => (2 x m) in our example
    % X' * error => (n x m) * (m x 1)
    theta = theta - (alpha/m) * (X' * error);





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
