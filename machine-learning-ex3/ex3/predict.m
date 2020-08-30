function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X is 5000 training examples and 400 values, each value representing a pixel in a 20x20 image
% Need to add bias unit to X so that we get 500 x 401
% X => 5000 x 400
% a1 = [ones;X] = (5000 x 1) , (5000 x 400) => (5000 x 401)
a1 = [ones(rows(X),1), X];
% Theta1 is matrix of (num nodes in layer 2) x (number of nodes in layer 1)
% Say 401 nodes in layer 1 because we consider each of 400 pixels an input node and we added bias unit
% Theta1 => (25 x 401)

% a1 => (5000 x 401)
% Theta1 => (25 x 401)
% Theta1' => (401 x 25)
% z2 = a1 * Theta1' => (5000 x 401) * (401 x 25) => (5000 x 25)
z2 = a1 * Theta1';
a2 = sigmoid(z2);
% Need to add bias unit to z2 to get a2
% z2 => (5000 x 25)
% ones(rows(z2),1) => (5000 x 1)
% a2 => (5000, 26)
a2 = [ones(rows(a2),1), a2];

% a2 => (5000 x 26)
% Theta2 = (10 x 26)
% Theta2 has 25 thetas for each node in the hidden layer (layer 2) and one extra for the bias unit making 26
% theta2' => (26 x 10)
% z3 => a2 * Theta2' => (5000 x 26) * (26 x 10) => (5000 x 10)
z3 = a2 * Theta2';
a3 = sigmoid(z3);

[prob, p] = max(a3, [], 2);









% =========================================================================


end
