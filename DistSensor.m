function s = DistSensor (pos, sig, N)
% Distance Sensor function
% Calculate the extracellular input for each cell
% Assuming distance function is squared Euclidean distance
% Input: 
%	[2,N]	: pos : position
% 	[3,N]	: sig : signal
% 	scalar	: N   : cell count
% Output: 
% 	[3,N]	: sensor

	k = 2; % Spatial decay constant -- See DEM.m from SPM12 toolkit
	d = pdist(pos', 'squaredeuclidean');
	s = sig * (exp(-k * squareform(d)) - eye(N));
end