function d_mu = InternalStatesUpdate (p_x, p_y, eps_x, eps_y, s_mu, N)
% Internal States Update function
%
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020
%
%
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
	
	% d_mu = k_mu * sigma'(mu) * (p_x * eps_x + p_y * eps_y)
	d_mu = zeros(3,N);
	for i = 1:N
		invSoftmax = (diag(s_mu(:,i)) - (s_mu(:,i)*s_mu(:,i)'));
		d_mu(:,i) = invSoftmax * (p_x * eps_x(:,i) + p_y * eps_y(:,i));
	end
end