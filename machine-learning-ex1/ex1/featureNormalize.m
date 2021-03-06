function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% X => (m x n) => (m x 2)
% Calculate the mean for each training example in m
% Using mean and std will calculate the mean and 
% standard deviation for columns of the given matrix
mu = mean(X);
sigma = std(X);
% Apply normalization
% (x - mu) / sigma
% Need to element-wise subtract mu from X
% Actually the code `X - mu` will work, but it's probably
% safer to ensure element-wise subtraction
% then divide by sigma
% X => (m x n)
% mu => (1 x n)
% sigma = (1 x n)
X_norm = (X .- mu) ./ sigma;







% ============================================================

end
