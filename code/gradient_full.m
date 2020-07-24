%% Simulation parameters
clear
clc

% Number of cells
N = 50;

% Time step size
dt = 0.01;
tLimit = 100;

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
psi_y = randn(3,N);

% [3,1] vector | Intracellular sensor
s_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular sensor
s_a = zeros(3,N); % Init values doesn't matter

% [3,1] vector | Intracellular prediction error
epsilon_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular prediction error
epsilon_a = zeros(3,N); % Init values doesn't matter


figure(1)
clf
cmap = mu';
h = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | Time: 0", N));
daspect([1 1 1])
axis([-1 1 -1 1])
grid on
xmax_prev = 1;
ymax_prev = 1;


%% Simulation loop
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
	
	% Plot
	try
		h.XData = psi_x(1,:);
		h.YData = psi_x(2,:);
		h.CData = mu';
		ht.String = sprintf("N: %d | Time: %.2f", N, t*dt);
		
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