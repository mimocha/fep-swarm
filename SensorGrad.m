function gradient = SensorGrad (pos, sec, C, N)
% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: position
% 	[C,N]	: secretion
% 	scalar	: C
% 	scalar	: N
% Output: 
% 	[2,C,N]	: gradient

	% [X,Y] are [j,i] matrices
	X = repmat(pos(1,:), [N,1]);
	Y = repmat(pos(2,:), [N,1]);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Pairwise Exponential Distance Decay Matrix
	% (From normal S_Alpha calculations)
	dd = pdist(pos', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([C,N] * ([N,N] .* [N,N]))
	dSdx = -2 .* k .* sec * (dx .* dd); 
	dSdy = -2 .* k .* sec * (dy .* dd);
	
	% Gradient matrix, [2,C,N]
	gradient = zeros(2,C,N);
	for i = 1:N
		gradient(1,:,i) = dSdx(:,i)';
		gradient(2,:,i) = dSdy(:,i)';
	end
end