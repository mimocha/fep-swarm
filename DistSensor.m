function S = DistSensor (pos, sec, N)
% Distance Sensor function
% Calculate the extracellular input for each cell
% Assuming distance function is squared Euclidean distance
% Input: 
%	[2,N]	: position
% 	[C,N]	: secretion
% 	scalar	: N
% Output: 
% 	[C,N]	: sensor

	% Spatial decay constant -- See DEM.m
	k = 2;
	d = pdist(pos', 'squaredeuclidean');
	S = sec * (exp(-k * squareform(d)) - eye(N));
end