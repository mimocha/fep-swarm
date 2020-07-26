% Mu Update Calculation function
% Calculates the change to mu
% Input: 
%	[3,3]	: prior
% 	[3,3,N] : d_sigma
% 	[3,N]	: epsilon
% Output: 
% 	[3,N]	: d_mu
function d_mu = MuUpdate(prior, d_sigma, epsilon, N)
	d_mu = zeros(3,N);
	
	% Iterate through each cell
	for i = 1:N
		d_mu(:,i) = -(prior * d_sigma(:,:,i)) * epsilon(:,i);
	end
end
