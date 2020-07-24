% Derivative of Softmax function
% Returns a Jacobian matrix of the softmax
% d_sigma = diag(mu) - (mu*mu')
% Input:  [3,N]	: mu
% Output: [3,3,N] : d_sigma
function d_sigma = DeriveSoftmax(mu, N)
	d_sigma = zeros(3,3,N);
	
	% Iterate through each cell
	for i = 1:N
		d_sigma(:,:,i) = diag(mu(:,i)) - (mu(:,i)*mu(:,i)');
	end
end