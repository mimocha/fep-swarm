%% Simulation parameters
clear
clc

GIF = false;
filename = 'Gradient.gif';

% Number of cells
N = 14;

% Time step size
dt = 0.05;
tLimit = 100;

%% Cell properties
% Generate N random cells

% Belief prior
prior = [1 1 0; ...
		 1 1 1; ...
		 0 1 1];

% [2,1] vector | Cell position
psi_x = randn(2,N);
% psi_x = [ cos(0:2*pi/N:2*pi) ; sin(0:2*pi/N:2*pi) ];
% psi_x(:,end) = [];
% [3,1] vector | Cell secretion
psi_y = randn(3,N);

% [3,1] vector | Sensor
S = zeros(3,N);

% [3,1] vector | Beliefs
% Diverges to infinity if sum of any column is greater than 1
mu = softmax(rand(3,N));

% [3,1] vector | Prediction error
epsilon = zeros(3,N);

% [3,1] vector | Chemical propagation coefficient
grad_S = zeros(2,3,N);



figure(1)
cmap = mu';
h1 = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht1 = title(sprintf("N: %d | Time: 0", N));
daspect([1 1 1])
axis([-1 1 -1 1])
grid on
xmax_prev = 1;
ymax_prev = 1;

% GIF
if GIF
	fig = gcf;
	SaveGIF(fig, filename, 'LoopCount', inf);
end

%% Simulation loop
for t = 1:tLimit/dt
	% 1. Sensory Inputs
	S = Alpha(psi_x, psi_y);

	% 2. Softmax Function
	sigma_mu = softmax(mu);
	
	% 3. Perception Error
	epsilon = S - (prior * sigma_mu);
	
	% 4. Update Position-Secretion
	da_y = -epsilon;
	psi_y = psi_y + (dt .* da_y);
	
	% Chemical Gradients
	grad_S = DeriveAlpha(psi_x, S, N);
	da_x = DxUpdate(eye(3), grad_S, epsilon, N);
	psi_x = psi_x + (dt .* da_x);
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	d_mu = MuUpdate(eye(3), d_sigma, epsilon, N);
	mu = mu + (dt .* d_mu);
	
	% Plot
	try
		h1.XData = psi_x(1,:);
		h1.YData = psi_x(2,:);
		h1.CData = mu';
		ht1.String = sprintf("N: %d | Time: %.2f", N, t*dt);
		
		% Update axis limit for visibility
		xmax = ceil(max(abs(psi_x(1,:)))/5);
		ymax = ceil(max(abs(psi_x(2,:)))/5);
		if (xmax_prev ~= xmax) || (ymax_prev ~= ymax)
			xmax_prev = xmax;
			ymax_prev = ymax;
			axis([-xmax xmax -ymax ymax]*5)
		end
		
		if GIF && (mod(t,4) == 0)
			drawnow
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		else
			drawnow
		end
	catch
		break;
	end
	
	Debug("Y", psi_y)
end



%% Functions

% Distance function
% Returns [3,N] matrix
function S = Alpha (psi_x, psi_y)
	S = psi_y * (exp(-squareform(pdist(psi_x'))));
end
