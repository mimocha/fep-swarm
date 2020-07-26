%% Simulation parameters
clear
clc

% Drawing interval (Number of sim step between drawing)
drawInt = 10;
% Axis display range
axRange = 3;
% Axis Lock? (Locks center of axis to cell cluster center)
axLock = true;
% Heatmap Grid Spacing
hmSpace = 0.1;

% Number of cells / Starting States
% Comment these 3 out, and set N directly to have randomly set starting mu
Nr = 9; % Red
Ng = 6; % Green
Nb = 1; % Blue
N = Nr + Ng + Nb;

% Time step size
dt = 0.001;
tLimit = 30;

% Anonymous dt Update function
DtUpdate = @(x,dx) x + (dt.*dx);

%% Cell properties

% Intracellular belief prior
prior_y = eye(3);
% Extracellular belief prior
prior_a =  [1 1 0; ...
			1 1 1; ...
			0 1 1];

% ======================== [3,1] vector | Beliefs ========================= %
% Diverges to infinity if sum of any column is greater than 1
try
	mu = [repmat([1;0;0],1,Nr) , repmat([0;1;0],1,Ng) , repmat([0;0;1],1,Nb)];
catch
	mu = randn(3,N);
end
mu = softmax(mu);

% ====================== [2,1] vector | Cell position ====================== %
psi_x = randn(2,N);

% ==================== [3,1] vector | Cell secretion ==================== %
psi_y = zeros(3,N);

% ==================== Sensor ==================== %
% [3,1] vector | Intracellular sensor
s_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular sensor
s_a = zeros(3,N); % Init values doesn't matter

% ==================== Prediction Error ==================== %
% [3,1] vector | Intracellular prediction error
epsilon_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular prediction error
epsilon_a = zeros(3,N); % Init values doesn't matter


%% Heatmap Mesh Grid
R = -axRange:hmSpace:axRange;
[X,Y] = meshgrid(R, R);

% Heatmap function
HMShape = @(hmap, X) reshape(hmap,size(X));
sig_maps = Heatmap (X, Y, psi_x, psi_y);
hmap1 = HMShape(sig_maps{1},X);
hmap2 = HMShape(sig_maps{2},X);
hmap3 = HMShape(sig_maps{3},X);


%% Figure setup
figure(1)
colormap jet
cmap = mu';

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready. Press key to begin.", N, dt));
grid on
hold on
hquiv = quiver(psi_x(1,:), psi_x(2,:), zeros(1,N), zeros(1,N), 'k');

% Heatmap Mu 1
ax2 = subplot(2,2,2);
hmu1 = pcolor(X,Y,hmap1);
title("\mu_1 (Red)")
grid on

% Heatmap Mu 2
ax3 = subplot(2,2,3);
hmu2 = pcolor(X,Y,hmap2);
title("\mu_2 (Green)")
grid on

% Heatmap Mu 3
ax4 = subplot(2,2,4);
hmu3 = pcolor(X,Y,hmap3);
title("\mu_3 (Blue)")
grid on

% Axis Tracking variables
axT = axRange;
axB = -axRange;
axL = -axRange;
axR = axRange;

% Styling
SetAll = @(H, propName, propVal) set(H, propName, propVal);
SetAll([ax1,ax2,ax3,ax4], 'DataAspectRatio', [1 1 1])
SetAll([ax1,ax2,ax3,ax4], 'XLim', [axL axR])
SetAll([ax1,ax2,ax3,ax4], 'YLim', [axB axT])
SetAll([ax1,ax2,ax3,ax4], 'CLim', [0 1])
SetAll([hmu1,hmu2,hmu3], 'EdgeColor', 'None')


%% Simulation loop

fprintf("Ready.\n")
pause

for t = 1:tLimit/dt
	%% Main Inference
	% 1. Sensory Inputs
	% Intracellular Sensor
	s_y = psi_y;
	% Extracellular Sensor
	s_a = Alpha(psi_x, psi_y, N);

	% 2. Softmax Function
	sigma_mu = softmax(mu);
	
	% 3. Perception Error
	epsilon_y = s_y - (prior_y * sigma_mu);
	epsilon_a = s_a - (prior_a * sigma_mu);
	
	% 4.1 Update Secretion
	da_y = -epsilon_y;
	psi_y = DtUpdate(psi_y, da_y);
	
	% 4.2 Update Position
	grad_S = DeriveAlpha(psi_x, s_a, N);
	da_x = DxUpdate(eye(3), grad_S, epsilon_a, N);
	psi_x = DtUpdate(psi_x, da_x);
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	d_mu = MuUpdate(eye(3), d_sigma, epsilon_a, N);
	mu = DtUpdate(mu, d_mu);
	
	% Draw on intervals only
	if mod(t,drawInt) ~= 0
		continue
	end
	
	%% Signal Mapping Function
	sig_maps = Heatmap (X, Y, psi_x, psi_y);
	hmap1 = HMShape(sig_maps{1},X);
	hmap2 = HMShape(sig_maps{2},X);
	hmap3 = HMShape(sig_maps{3},X);
	
	%% Plot
	try
		% Update Cell Scatter
		SetAll(hmain, {'XData','YData','CData'}, {psi_x(1,:),psi_x(2,:),mu'})
		ht.String = sprintf("N: %d | dt: %.3f | Time: %.3f", N, dt, t*dt);
		
		% Update Heatmaps
		SetAll([hmu1;hmu2;hmu3], {'CData'}, {hmap1;hmap2;hmap3});

		% Update cell center to be axis center
		if axLock
			psi_x = psi_x - mean(psi_x,2);
			
			% Position change relative to overall cluster movement
			da_x = da_x - mean(da_x,2);
		end
		
		% Update Cell Quiver
		SetAll(hquiv, {'XData','YData','UData','VData'}, ...
			{psi_x(1,:),psi_x(2,:),da_x(1,:),da_x(2,:)})
		
		drawnow
	catch ME
		warning("Drawing loop broken. Error given: '%s'", ME.message)
		break
	end
end


%% Functions

% Distance function
% Calculate the sensory input for each cell
% Assuming distance function is squared Euclidean distance
% Input: 
%	[2,N]	: psi_x
% 	[3,N]	: psi_y
% 	scalar	: N
% Output: 
% 	[3,N]	: s_a
function s_a = Alpha (psi_x, psi_y, N)
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	if N > 1
		d = pdist(psi_x', 'squaredeuclidean');
		s_a = psi_y * (exp(-k * squareform(d)) - eye(N));
	else
		s_a = zeros(3,1);
	end
end

% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: psi_x
% 	[3,N]	: S
% 	scalar	: N
% Output: 
% 	[2,3,N]	: grad_S
function grad_S = DeriveAlpha (psi_x, S, N)
	% [X,Y] are [j,i] matrices
	X = repmat(psi_x(1,:), [N,1]);
	Y = repmat(psi_x(2,:), [N,1]);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = -2 .* k .* (X - X'); % -2 * k * (x_j - x_i) | [N,N]
	dy = -2 .* k .* (Y - Y'); % -2 * k * (y_j - y_i) | [N,N]
	
	% Final partial derivative step
	dSdx = S * dx; % [3,N] = [3,N] * [N,N]
	dSdy = S * dy; % [3,N] = [3,N] * [N,N]
	
	% Gradient matrix, [x x x ; y y y]
	grad_S = zeros(2,3,N);
	for i = 1:N
		grad_S(1,:,i) = dSdx(:,i)';
		grad_S(2,:,i) = dSdy(:,i)';
	end
end

% Dx Update Calculation function
% Calculates the change to dx
% Input: 
%	[3,3]	: prior
% 	[3,3,N] : grad_S
% 	[3,N]	: epsilon
% Output: 
% 	[3,N]	: da_x
function da_x = DxUpdate(prior, grad_S, epsilon, N)
	da_x = zeros(2,N);
	
	% Iterate through each cell
	% Essentially, each cell gets its own [2,3] "gradient matrix", which
	% modulates how `epsilon_a` is "perceived" in each direction, thus
	% modulating how the cell moves.
	for i = 1:N
		% [2,1]   =  ([2,3,1]		* [3,3]) * [3,1]
		da_x(:,i) = -(grad_S(:,:,i) * prior) * epsilon(:,i);
	end
end

% Derivative of Softmax function
% Returns a Jacobian matrix of the softmax
% d_sigma = diag(mu) - (mu*mu')
% Input:  
% 	[3,N]	: mu
% 	scalar	: N
% Output: 
%	[3,3,N] : d_sigma
function d_sigma = DeriveSoftmax(mu, N)
	d_sigma = zeros(3,3,N);
	
	% Iterate through each cell
	for i = 1:N
		% [3,3,1]	   = [3,3]		   - ([3,1]  *  [1,3])
		d_sigma(:,:,i) = diag(mu(:,i)) - (mu(:,i)*mu(:,i)');
	end
end

% Mu Update Calculation function
% Calculates the change to mu
% Input: 
%	[3,3]	: prior (identity matrix)
% 	[3,3,N] : d_sigma
% 	[3,N]	: epsilon
%	scalar	: N
% Output: 
% 	[3,N]	: d_mu
function d_mu = MuUpdate(prior, d_sigma, epsilon_a, N)
	d_mu = zeros(3,N);
	
	% Iterate through each cell
	for i = 1:N
		% [3,1]   = ([3,3] * [3,3,1])		 * [3,1]
		d_mu(:,i) = -(prior * d_sigma(:,:,i)) * epsilon_a(:,i);
	end
end

% Heatmap Calculations
% Input:
%	[P,P]	: X | x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y | y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x | cell coordinates
%	[3,N]	: psi_y | cell chemical signals
% Output:
%	{[P*P,1], [P*P,1], [P*P,1]} : sig_maps | signal heatmap for each signal type
function sig_maps = Heatmap (X, Y, psi_x, psi_y)
	% Distance from each cell to each reference point in each dimensions
	x_diff = repmat(psi_x(1,:), numel(X), 1) - X(:);
	y_diff = repmat(psi_x(2,:), numel(Y), 1) - Y(:);
	
	% Distance function (exponential decay of squared euclidean distance)
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