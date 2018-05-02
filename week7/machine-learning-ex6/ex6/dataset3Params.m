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
Cpara = [0.01;0.03;0.1;0.3;1;3;10;30];
sigmapara = [0.01;0.03;0.1;0.3;1;3;10;30];
error=zeros(size(Cpara,1),size(sigmapara,1));
for i=1:size(Cpara,1)
    for j=1:size(sigmapara,1)
        C=Cpara(i);
        sigma=sigmapara(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error(i,j)=mean(double(predictions ~= yval));
    end
end

[position1,position2]=find(error==min(min(error)));  %找到矩阵A中的最小值位置
C=Cpara(position1);
sigma=sigmapara(position2);








% =========================================================================

end
