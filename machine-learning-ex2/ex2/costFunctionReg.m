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


h = @(x) sigmoid( sum( theta'.*x , 2 ) ); x = X; % to clean up bottom code

J = (1/m) * sum( -y.*log( h(x) ) - (1-y).*log( 1-h(x) ) ) + (lambda/(2*m))*( sum(theta.^2) - theta(1).^2 ); % by convention we do not normalize the theta_0

grad = (1/m) * sum( ( h(x)-y ) .* x ) + lambda/m * (  [ 0 theta( 2: size(theta',2) )'] ) ; % complicated expression is theta returned with first element set to zero




% =============================================================

end