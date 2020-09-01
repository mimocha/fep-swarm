function grad = SensorGrad (pos, sig, N)
% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: pos : cell position
% 	[3,N]	: sig : cell signal
% 	scalar	: N   : cell count
% Output: 
% 	[2,3,N]	: gradient

	% [X,Y] are [j,i] matrices
	X = repmat(pos(1,:), [N,1]);
	Y = repmat(pos(2,:), [N,1]);
	
	% Pairwise Exponential Distance Decay Matrix
	k = 2; % Spatial decay constant -- See DEM.m from spm12 toolkit
	dd = pdist(pos', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([3,N] * ([N,N] .* [N,N]))
	dsdx = -2 .* k .* sig * (dx .* dd); 
	dsdy = -2 .* k .* sig * (dy .* dd);
	
	% Gradient matrix, [2,3,N]
	grad = zeros(2,3,N);
	for i = 1:N
		grad(1,:,i) = dsdx(:,i)';
		grad(2,:,i) = dsdy(:,i)';
	end
end