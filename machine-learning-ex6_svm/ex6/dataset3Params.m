function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % vals from which optimal C and sigma is chosen
l = length(vals);
errors = zeros(l); % rows are C vals, columns are sigma vals

for i=1:l
    for j=1:l
        Ci = vals(i);
        sigmaj = vals(j);
        model = svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigmaj));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        errors(i,j) = error;
    end
end
minError = min(min(errors));
[x,y] = find(errors == minError);
C = vals(x);
sigma = vals(y);



% =========================================================================

