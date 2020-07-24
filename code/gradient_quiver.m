%% Simulation parameters
clear
clc

% Number of cells
N = 25;

% Time step size
dt = 0.01;
tLimit = 10;

%% Cell properties
% Generate N random cells

% Intracellular belief prior
prior_y = eye(3);
% Extracellular belief prior
prior_a =  [1 1 0; ...
			1 1 1; ...
			0 1 1];

% [3,1] vector | Beliefs
% Diverges to infinity if sum of any column is greater than 1
mu = softmax(randn(3,N));

% [2,1] vector | Cell position
psi_x = randn(2,N);
% [3,1] vector | Cell secretion
psi_y = softmax(mu);

% [3,1] vector | Intracellular sensor
s_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular sensor
s_a = zeros(3,N); % Init values doesn't matter

% [3,1] vector | Intracellular prediction error
epsilon_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular prediction error
epsilon_a = zeros(3,N); % Init values doesn't matter

% Quiver Plot Mesh Grid
R = -25:1:25; % Range of gradient arrows to calculate
[X,Y] = meshgrid(R, R);

% Generative model g(mu_i)
G = {repmat(reshape(prior_a*softmax([1,0,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(prior_a*softmax([0,1,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(prior_a*softmax([0,0,1]'), [1,1,3]), size(X))};

[U,V] = GradientMap (G, X, Y, psi_x, psi_y);

figure(1)
cmap = mu';
h = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | Time: 0", N));
hold on
grid on
daspect([1 1 1])
xmax_prev = 10;
ymax_prev = 10;
axis([-xmax_prev xmax_prev -ymax_prev ymax_prev])
hquiv1 = quiver(X,Y,U{1},V{1},'r');
hquiv2 = quiver(X,Y,U{2},V{2},'g');
hquiv3 = quiver(X,Y,U{3},V{3},'b');


%% Simulation loop

fprintf("Ready. Press any key to begin.\n")
pause

for t = 1:tLimit/dt
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
	
	% 4. Update Position-Secretion
	da_y = -epsilon_y;
	psi_y = psi_y + (dt .* da_y);
	
	% Chemical Gradients
	grad_S = DeriveAlpha(psi_x, s_a, N);
	da_x = DxUpdate(eye(3), grad_S, epsilon_a, N);
	psi_x = psi_x + (dt .* da_x);
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	d_mu = MuUpdate(eye(3), d_sigma, epsilon_a, N);
	mu = mu + (dt .* d_mu);
	
	% Gradient Mapping Function
	[U,V] = GradientMap (G, X, Y, psi_x, psi_y);
	
	% Plot
	try
		h.XData = psi_x(1,:);
		h.YData = psi_x(2,:);
		h.CData = mu';
		ht.String = sprintf("N: %d | Time: %.2f", N, t*dt);
		
		% Update Quiver Plots
		hquiv1.UData = U{1};
		hquiv1.VData = V{1};
		hquiv2.UData = U{2};
		hquiv2.VData = V{2};
		hquiv3.UData = U{3};
		hquiv3.VData = V{3};
		
		% Update axis limit for visibility
		xmax = ceil(max(abs(psi_x(1,:)))/5);
		ymax = ceil(max(abs(psi_x(2,:)))/5);
		if (xmax_prev ~= xmax) || (ymax_prev ~= ymax)
			xmax_prev = xmax;
			ymax_prev = ymax;
			axis([-xmax xmax -ymax ymax]*5)
		end
		
		drawnow
	catch
		break
	end
	
	% Break if Mu becomes NaN
	if any(isnan(mu),'all')
		ht.String = sprintf("N: %d | Time: %.2f | Diverged!", N, t*dt);
		fprintf("Mu diverged!\n")
		disp(mu)
		break
	end
end



%% Functions

% Distance function
function s_a = Alpha (psi_x, psi_y, N)
	% Default Euclidean distance assumption
	% Returns [3,N] matrix
	% Dimensions will mismatch step 4, position update
	s_a = psi_y * (exp(-squareform(pdist(psi_x'))) - eye(N));
end

% Gradient function
% Returns [2,3,N] matrix
function grad_S = DeriveAlpha (psi_x, S, N)
	X = repmat(psi_x(1,:), [N,1]);
	Y = repmat(psi_x(2,:), [N,1]);
	
	% Partial Derivatives w.r.t. x/y
	dg = squareform(pdist(psi_x')); % Euclidean distance | [N,N]
	dx = -(X' - X) ./ dg; % -(x_i - x_j) ./ (sqrt(x^2 + y^2)) | [N,N]
	dy = -(Y' - Y) ./ dg; % -(y_i - y_j) ./ (sqrt(x^2 + y^2)) | [N,N]
	
	% Set diagonals to zeros (normally NaNs because division by zeros)
	dx(eye(N)==1) = 0;
	dy(eye(N)==1) = 0;
	
	% Final partial derivative step
	dSdx = S * dx; % [3,N] = [3,N] * [N,N]
	dSdy = S * dy; % [3,N] = [3,N] * [N,N]
	
	% Gradient matrix: [x1, x2, x3; y1, y2, y3]
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
		d_mu(:,i) = (prior * d_sigma(:,:,i)) * epsilon_a(:,i);
	end
end

% Gradient Quiver Calculations
% Input:
%	{[P,P,3], [P,P,3], [P,P,3]}	: G (fixed generative model g(mu))
%	[P,P]	: X | x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y | y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x | cell coordinates
%	[3,N]	: psi_y | cell chemical signals
% Output:
%	{[P,P,3], [P,P,3], [P,P,3]} : U | x-component of gradient vector
%	{[P,P,3], [P,P,3], [P,P,3]} : V | y-component of gradient vector
function [U, V] = GradientMap (G, X, Y, psi_x, psi_y)
	% Distance from each cell to each reference point in each dimensions
	x_diff = repmat(psi_x(1,:), numel(X), 1) - X(:);
	y_diff = repmat(psi_x(2,:), numel(Y), 1) - Y(:);
	
	% Distance function (exponential decay of euclidean distance)
	euc_dist = sqrt(x_diff.^2 + y_diff.^2);
	dist_decay = exp(-euc_dist);
	
	% Decay of each signal across grid -- Signal Heatmap
	mu1_decay = sum(dist_decay.*psi_y(1,:),2);
	mu2_decay = sum(dist_decay.*psi_y(2,:),2);
	mu3_decay = sum(dist_decay.*psi_y(3,:),2);
	
	% Concatenate into 3D matrix of S_alpha & reshape to match grid
	s_alpha_grid = cat(3, mu1_decay, mu2_decay, mu3_decay);
	s_alpha_grid = reshape(s_alpha_grid, [size(X),3]);
	
	% Epsilon Alpha Calculations
	epsilon_mu1 = s_alpha_grid - G{1}; % [P,P,3]
	epsilon_mu2 = s_alpha_grid - G{2}; % [P,P,3]
	epsilon_mu3 = s_alpha_grid - G{3}; % [P,P,3]
	
	% Derivative Alpha Calculations
	d_alpha_x = -x_diff ./ euc_dist;
	d_alpha_y = -y_diff ./ euc_dist;
	% NaN Check
	d_alpha_x(isnan(d_alpha_x)) = 0;
	d_alpha_y(isnan(d_alpha_y)) = 0;
	% Reshape
	d_alpha_x = reshape(d_alpha_x, [size(X),size(d_alpha_x,2)]);
	d_alpha_y = reshape(d_alpha_y, [size(Y),size(d_alpha_y,2)]);
	
	% Final Partial Derivative Step
	% Mu 1
	psi1_dSdx = sum(d_alpha_x .* s_alpha_grid(:,:,1), 3); % [P,P]
	psi1_dSdy = sum(d_alpha_y .* s_alpha_grid(:,:,1), 3); % [P,P]
	% Mu 2
	psi2_dSdx = sum(d_alpha_x .* s_alpha_grid(:,:,2), 3); % [P,P]
	psi2_dSdy = sum(d_alpha_y .* s_alpha_grid(:,:,2), 3); % [P,P]
	% Mu 3
	psi3_dSdx = sum(d_alpha_x .* s_alpha_grid(:,:,3), 3); % [P,P]
	psi3_dSdy = sum(d_alpha_y .* s_alpha_grid(:,:,3), 3); % [P,P]
	
	% Combine into propagation matrix
	grid_dx = cat(3, psi1_dSdx, psi2_dSdx, psi3_dSdx);
	grid_dy = cat(3, psi1_dSdy, psi2_dSdy, psi3_dSdy);
	
	% Change to dot_A_x
	% MU_A = dSxy .* epsilon
	% [P,P] = -sum( [P,P,3] .* [P,P,3] , 3 )
	% Mu 1
	mu1_ax = -sum(grid_dx .* epsilon_mu1, 3);
	mu1_ay = -sum(grid_dy .* epsilon_mu1, 3);
	% Mu 2
	mu2_ax = -sum(grid_dx .* epsilon_mu2, 3);
	mu2_ay = -sum(grid_dy .* epsilon_mu2, 3);
	% Mu 3
	mu3_ax = -sum(grid_dx .* epsilon_mu3, 3);
	mu3_ay = -sum(grid_dy .* epsilon_mu3, 3);
	
	% Combine and return UV vector map
	U = {mu1_ax, mu2_ax, mu3_ax}; % Change in X
	V = {mu1_ay, mu2_ay, mu3_ay}; % Change in Y
end