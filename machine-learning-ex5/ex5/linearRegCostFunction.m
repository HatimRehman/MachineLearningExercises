function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = @(x) sum( theta'.*x , 2 ) ; x = X; % to clean up bottom code

J = (1/(2*m)) * sum(( h(x)-y ).^2) + (lambda/(2*m))*( sum(theta.^2) - theta(1).^2 ); % by convention we do not normalize the theta_0

grad = (1/m) * sum( ( h(x)-y ) .* x ) + lambda/m * (  [ 0 theta( 2: size(theta',2) )'] ) ; % complicated expression is theta returned with first element set to zero









% =========================================================================

grad = grad(:);

end
