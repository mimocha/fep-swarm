function d_mu = InternalStatesUpdate (p_x, p_y, eps_x, eps_y, s_mu, N)
% Internal States Update function
% Calculates the change to internal states
% Input: 
% 	[3,3]	: p_x   : extracellular parameters
% 	[3,3]	: p_y   : intracellular parameters
% 	[3,N]	: eps_x : prediction error
% 	[3,N]	: eps_y : prediction error
% 	[3,N]	: s_mu  : belief
% 	scalar	: N     : cell count
% Output: 
% 	[3,N]	: d_mu  : internal states update
	
	% d_mu = -(Px + Py) * sigma'(mu) * (eps_x + eps_y)
	error = eps_x + eps_y;
	params = p_x + p_y;
	d_mu = zeros(3,N);
	for i = 1:N
		% Inverse softmax
		invSoftmax = (diag(s_mu(:,i)) - (s_mu(:,i)*s_mu(:,i)'));
		d_mu(:,i) = - params * invSoftmax * error(:,i);
	end
end