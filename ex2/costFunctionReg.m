function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
J = sum(-y .* log(h) - (1-y) .* log(1-h))/m + lambda * norm(theta(2:length(theta)))^2/2/m;
% dot product (h(x)-y) * x for every dimision of x
% a matrix of same column of h-y used for convenience of calculation
h_minus_y = repmat((h - y), 1, size(X,2));
% matrix form
grad = transpose(dot(h_minus_y, X, 1)) / m + lambda * theta / m;
grad(1) = grad(1) - lambda * theta(1) /m;
% use loop, easier to understand
% grad(1) = (h - y)' * X(:,1) /m;
% for j = 2:length(theta)
%     grad(j) = (h - y)' * X(:,j) /m + lambda * theta(j) /m;
% end




% =============================================================

end
