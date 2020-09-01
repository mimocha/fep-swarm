%% Parameter Initialization
% Simulation Parameters
% tLimit :: simulation time limit
% dt     :: integration time step
% N      :: number of cells
%
% Variable Definitions
%	p_x    :: extracellular model parameter
%	p_y    :: intracellular model parameter
%	mu     :: internal state
%	s_mu   :: belief -- softmax(mu)
%	psi_x  :: position
%	psi_y  :: chemical signal
%	s_x    :: extracellular sensor
%	s_y    :: intracellular sensor
%	a_x    :: action (movement)
%	a_y    :: action (signal change)
%	k_a    :: learning rate (action)
%	k_mu   :: learning rate (belief)

%% Simulation Loop
for t = 1:tLimit/dt
	% Sensory Inputs
	s_y = psi_y + Noise(N);
	s_x = DistanceFunc(psi_x, psi_y, N) + Noise(N);
	
	% Generative Model
	s_mu = softmax(mu);
	g_x = (p_x * s_mu);
	g_y = (p_y * s_mu);
	
	% Sensory-Prediction Error
	eps_x = s_x - g_x;
	eps_y = s_y - g_y;
	
	% Calculate Action
	grad = GradientFunc(psi_x, psi_y, N);
	a_x = -grad * eps_x;
	a_y = -eps_y;
	
	% Update World
	psi_x = psi_x + (dt .* k_a .* a_x);
	psi_y = psi_y + (dt .* k_a .* a_y);
	
	% Internal State Update
	d_mu = InternalStateFunc(p_x, p_y, eps_x, eps_y, s_mu);
	mu = mu + (dt .* k_mu .* d_mu);
end

%% Helper Functions
function omega = Noise(N)
% Noise Generation Function
	omega = sqrt(1/exp(16)) * randn([3,N]);
end

function s_x = DistanceFunc(psi_x, psi_y, N)
	% Pairwise squared Euclidean distance between cells
	dd = squareform(pdist(psi_x', 'squaredeuclidean'));
	% Exponential decay modulates signal strength
	s_x = psi_y * (exp(-2 .* dd) - eye(N));
end

function grad = GradientFunc(psi_x, psi_y, N)
	X = repmat(psi_x(1,:), [N,1]);
	Y = repmat(psi_x(2,:), [N,1]);
	% Signal Decay
	dd = squareform(pdist(psi_x', 'squaredeuclidean'));
	dd = exp(-2 .* dd) - eye(N);
	% Calculate Partial Derivatives
	dx = X - X';
	dy = Y - Y';
	dsdx = -4 .* psi_y * (dx .* dd); 
	dsdy = -4 .* psi_y * (dy .* dd);
	% Gradient matrix
	grad = zeros(2,3,N);
	for i = 1:N
		grad(1,:,i) = dsdx(:,i)';
		grad(2,:,i) = dsdy(:,i)';
	end
end

function d_mu = InternalStateFunc(p_x, p_y, eps_x, eps_y, s_mu, N)
	epsilon = eps_x + eps_y;
	p_sum = p_x + p_y;
	d_mu = zeros(C,N);
	for i = 1:N
		inv_mu = (diag(s_mu(:,i)) - (s_mu(:,i)*s_mu(:,i)'));
		d_mu(:,i) = - p_sum * inv_mu * epsilon(:,i);
	end
end