function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Finds positive and negative indecies
% ex: The first index, 1, is classified as 0, so neg holds 1 as it's first element in
%       the vector

pos = find(y == 1);
neg = find(y == 0);
plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 2, ...
        'MarkerSize', 7);
plot(X(neg,1), X(neg,2), 'ko', 'MarkerFaceColor', 'y', ...
        'MarkerSize', 7);







% =========================================================================



hold off;

end
