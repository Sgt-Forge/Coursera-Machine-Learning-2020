function A = warmUpExercise()
%WARMUPEXERCISE Example function in octave
%   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

A = [];
% ============= YOUR CODE HERE ==============
% Instructions: Return the 5x5 identity matrix 
%               In octave, we return values by defining which variables
%               represent the return values (at the top of the file)
%               and then set them accordingly.

% Returns identity matrix 5x5 
eye(5); 
% Returns identiry matrix 10x1
eye(10,1);
% Returns identity matrix 10x3
eye(10,3);
vec = [7,4];
% Returns identity matrix using two scalars of input vector
% Returns identity matrix 7x4
eye(vec);
% Update vector 1x1
vec = [5];
% Still works for single scalar
eye(vec);

A = eye(5);






% ===========================================


end
