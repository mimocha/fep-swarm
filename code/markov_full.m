%% Simulation parameters
clear
clc

% Number of cells
N = 25;
% Step Size
dt = 0.05;
% Time Limit
tLimit = 20;


%% Cell properties

% Intracellular belief prior
prior_y =  [1 0 0; ...
			0 1 0; ...
			0 0 1];
% Extracellular belief prior
prior_a =  [1 1 0; ...
			1 1 1; ...
			0 1 1];

% [2,1] vector | Cell position
psi_x = randn(2,N);
% [3,1] vector | Cell secretion
psi_y = randn(3,N);

% [3,1] vector | Intracellular sensor
s_y = zeros(3,N);
% [3,1] vector | Extracellular sensor
s_a = zeros(3,N);

% [3,1] vector | Beliefs
% Diverges to infinity if sum of any column is greater than 1
mu = softmax(rand(3,N));

% [3,1] vector | Intracellular prediction error
epsilon_y = zeros(3,N);
% [3,1] vector | Extracellular prediction error
epsilon_a = zeros(3,N);

% [3,1] vector | Chemical propagation coefficient
% Used to convert epsilon_a to size [2,N] for updating cell position
kai = ones(2,3);


figure(1)
cmap = mu';
h = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | Time: 0", N));


%% Simulation loop
for t = 1:tLimit/dt
	% 1. Sensory Inputs
	% Intracellular Sensor
	s_y = psi_y;
	% Extracellular Sensor
	% Subtract identity matrix from pairwise distance matrix
	s_a = psi_y * (exp(-squareform(pdist(psi_x'))) - eye(N));

	% 2. Softmax Function
	sigma_mu = softmax(mu);
	
	% 3. Perception Error
	epsilon_y = s_y - (prior_y * sigma_mu);
	epsilon_a = s_a - (prior_a * sigma_mu);
	
	% 4. Update Position-Secretion
	da_y = -epsilon_y;
	psi_y = psi_y + (dt .* da_y);
	
	% kai converts matrix from [3,N] to [2,N]
	da_x = -kai * epsilon_a;
	psi_x = psi_x + (dt .* da_x);
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	% Use identity matrix as prior for now
	d_mu = MuUpdate(eye(3), d_sigma, epsilon_a, N);
	mu = mu + (dt .* d_mu);
	
	% Plot
	try
		h.XData = psi_x(1,:);
		h.YData = psi_x(2,:);
		h.CData = mu'; % Give the cells RGB color, based on mu.
		ht.String = sprintf("N: %d | Time: %.2f", N, t*dt);
		drawnow
	catch
		break
	end
end


%% Helper Functions

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


