function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X => (m x n) => (5000 x 400) => 5000 training and 400 features for each pixel
% Need to add bias
a1 = [ones(rows(X),1), X];
% a1 => (m x n) => (5000 x 401)
% Theta1 => (hidden_nodes x n) => (25 x 401).  Theta matrix is a row for each hidden node and a column for each input
% Theta1' => (n x hidden_nodes) => (401 x 25)
% z2 = sigmoid(a2 * Theta1') => (m x n) * (n x hidden_nodes) => (m x hidden_nodes)
z2 = sigmoid(a1 * Theta1');
% Add bias to a2
a2 = [ones(rows(z2),1), z2];
% a2 => (m x hidden_nodes) => (5000 x 26)
% Theta2 => (output_nodes x hidden_ndoes) => (10 x 26)
% Theta2' => (26 x 10)
% z3 => sigmoid(a2 * Theta2') => (5000 x 26) * (26 x 10) => (5000 x 10)
z3 = sigmoid(a2 * Theta2');
% Need y to be a matrix of 0s and 1s for multiclassification
% y => (5000 x 1)
% Below creates 5000 rows for each training example with 1 to num_lables in colums
% If the element at a column == y, then put 1, else put 0
y_Vec = (1:num_labels) == y;
% log(z3) => z3 => (5000 x 10)
% y_Vec = (5000 x 10)
% cost = y_Vec .* z3 => (5000 x 10) .* (5000 x 10) => (5000 x 10)
% sum(cost(:)) => scalar
cost = -y_Vec .* log(z3) - (1 .- y_Vec) .* log(1 - z3);
size(cost)
J = (1/m) * sum(cost(:));















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
