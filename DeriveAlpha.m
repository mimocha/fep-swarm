% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: psi_x
% 	[3,N]	: psi_y
% 	scalar	: N
% Output: 
% 	[2,3,N]	: grad_S
function grad_S = DeriveAlpha (psi_x, psi_y, N)
	% [X,Y] are [j,i] matrices
	X = repmat(psi_x(1,:), [N,1]);
	Y = repmat(psi_x(2,:), [N,1]);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Pairwise Exponential Distance Decay Matrix
	% (From normal S_Alpha calculations)
	dd = pdist(psi_x', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([3,N] * ([N,N] .* [N,N]))
	dSdx = -2 .* k .* psi_y * (dx .* dd); 
	dSdy = -2 .* k .* psi_y * (dy .* dd);
	
	% Gradient matrix, [x x x ; y y y]
	grad_S = zeros(2,3,N);
	for i = 1:N
		grad_S(1,:,i) = dSdx(:,i)';
		grad_S(2,:,i) = dSdy(:,i)';
	end
end