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

% feedforward computation for h_theta(x)
a1 = X;

a1 = [ones(m, 1) a1]; % add ones column

a2 = sigmoid( a1*Theta1'); 

a2 = [ones(m,1) a2 ]; % add ones column

a3 = sigmoid( Theta2*a2');

h = @(x) a3' ; x = X; % to clean up bottom code

% J_theta calculation
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :); % translate elements of y into a vector of 1's and 0's
end
 
J = (1/m) * sum(sum( -Y.*log( h(x) ) - (1-Y).*log( 1-h(x) ) ));

% calculate and add regularization to J_theta
Theta1_unbiased = Theta1(:, 2:end); %remove first column
Theta2_unbiased = Theta2(:, 2:end); %remove first column

regularization = (lambda/(2*m)) * (sum( sum(Theta1_unbiased.^2) ) + sum( sum(Theta2_unbiased.^2) ));

J = J + regularization;


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

%%%%%%%% Inefficient way %%%%%%%%%
% DELTA_1 = 0; DELTA_2 = 0;
% for i = 1:m
%     
%     %step 1: find the activation a_3 for the ith example using a feedforward pass
%     a_1 = X(i, :); % get ith row  
%     
%     a_1 = [1 a_1]; % add ones column   1x401
% 
%     a_2 = sigmoid( a_1*Theta1' );  
% 
%     a_2 = [1 a_2 ]; % add ones column   1x26
%     
%     a_3 = sigmoid( a_2*Theta2' ); 
%     
%     %step 2: For each output unit k in layer 3, set delta(3)_k 
%     delta_3 = a_3 - Y(i,:); % 1 x 10
%     
%     %step 3: set delta_2
%     
%     delta_2 = [(Theta2)'*delta_3']'.*sigmoidGradient( [1 a_1*Theta1'] );
%     
%     %step 4: set DELTA_l for this training example
%     
%     delta_2 = delta_2(2:end); % remove first element
%     %size(delta_2)
%     %size(a_1)
%     DELTA_1 = DELTA_1 + delta_2'*(a_1);
%     DELTA_2 = DELTA_2 + delta_3'*(a_2);
%     
% end

%%%%%%%% Efficient way %%%%%%%%%

d3 = a3' - Y; %size 5000x10

d2 = (d3*Theta2).* sigmoidGradient( [ones(m, 1) a1*Theta1'] );  % (5000x10 * 10x26).*(5000x401 * 401x25 + 1 column) = 5000x26
d2 = d2(:, 2:end); %remove first element 5000x25

D1 = d2'*a1; %25x401

D2 = d3'*a2; %26x10

Theta1_grad = (1/m)*D1 + (lambda/m)*[zeros(size(Theta1_unbiased,1), 1) Theta1_unbiased]; %add column of zeroes back to The
Theta2_grad = (1/m)*D2 + (lambda/m)*[zeros(size(Theta2_unbiased,1), 1) Theta2_unbiased];



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
