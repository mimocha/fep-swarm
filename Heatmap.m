% Heatmap Calculations
% Input:
%	[P,P]	: X | x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y | y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x | cell coordinates
%	[3,N]	: psi_y | cell chemical signals
% Output:
%	{[P,P,3], [P,P,3], [P,P,3]} : sig_maps | signal heatmap for each signal type
function sig_maps = Heatmap (X, Y, psi_x, psi_y)
	% Distance from each cell to each reference point in each dimensions
	x_diff = repmat(psi_x(1,:), numel(X), 1) - X(:);
	y_diff = repmat(psi_x(2,:), numel(Y), 1) - Y(:);
	
	% Distance function (exponential decay of euclidean distance)
	k = 2;
	euc_dist = x_diff.^2 + y_diff.^2;
	dist_decay = exp(-k .* euc_dist);
	
	% Decay of each signal across grid -- Signal Heatmap
	mu1 = sum(dist_decay.*psi_y(1,:),2);
	mu2 = sum(dist_decay.*psi_y(2,:),2);
	mu3 = sum(dist_decay.*psi_y(3,:),2);
	
	% Normalize to [0,1]
	% Relative strength of all signals, scaled to the max signal strength or 1,
	% whichever is higher.
	mu_max = max( [mu1;mu2;mu3;1] ,[] ,'all');
	norm = @(x,y) (x/y);
	mu1 = norm(mu1,mu_max);
	mu2 = norm(mu2,mu_max);
	mu3 = norm(mu3,mu_max);
	
	% Output
	sig_maps = {mu1, mu2, mu3};
end