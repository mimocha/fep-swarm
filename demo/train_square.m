%% Training Square Lattice Shape Demo
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020
%
% This file is for training square lattice shape (section 3.2.2 in the 
% dissertation, figures 3.5 -- 3.7)
% Drawing interval can be set really high to speed up process (nothing happens
% on-screen anyway).
%
% Used in conjunction with `test_square.m`



%% Simulation parameters
clear
clc

% Drawing interval
drawInt = 5000;
% Axis display range
axRange = 4;

% Number of cells
% Types of cells doesn't matter, but require a square number with odd roots.
N = 7^2;

% Time step size
dt = 0.01;
% Time limit
tLimit = 10000;
% Time
t = 0;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);

% Index of cells to print interesting variables to console, during training
idx = [1, 2, floor(N/2), ceil(N/2), N-1, N];



%% Cell properties
% =============== Learning Rates ============================================= %
k_p = 0.01;

% =============== Generative Parameter ======================================= %
% Random initialization
p_x = rand(3);
p_y = rand(3);

% =============== Internal States ============================================ %
% Bayer Filter (works only with odd square roots of N)
mu = BayerFilter(N);
	
% =============== Belief ===================================================== %
sigma_mu = exp(mu) ./ sum(exp(mu),1);

% =============== Cell Position ============================================== %
% Generate Square Lattice for N cells, with distance between cells of d
d = 1;
psi_x = SquareLattice(N, d);

% =============== Cell Signals =============================================== %
psi_y = softmax(mu);

% =============== Sensor States ============================================== %
s_x = zeros(3,N); % Extracellular
s_y = zeros(3,N); % Intracellular

% =============== Prediction Error =========================================== %
epsilon_x = zeros(3,N); % Extracellular
epsilon_y = zeros(3,N); % Intracellular



%% Figure setup
figure(1)
clf
cmap = sigma_mu';

% Scatter Plot
ax1 = gca;
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
titletext = sprintf("N: %d | k_p: %.2f | dt: %.2f | Time: %.2f", ...
				N, k_p, dt, dt*t);
ht = title(titletext);
grid on
hold on
xticks(-axRange:axRange)
yticks(-axRange:axRange)

% Axis Tracking variables
axT = axRange;
axB = -axRange;
axL = -axRange;
axR = axRange;

% Styling
SetAll = @(H, propName, propVal) set(H, propName, propVal);
SetAll(ax1, 'DataAspectRatio', [1 1 1])
SetAll(ax1, 'XLim', [axL axR])
SetAll(ax1, 'YLim', [axB axT])
SetAll(ax1, 'CLim', [0 1])



%% Training loop

fprintf("Ready. Press any key to begin ...\n")
pause

for t = 1:tLimit/dt
	% 1. Sensory Inputs
	s_x = DistSensor(psi_x, psi_y, N) + Noise(N);
	s_y = psi_y + Noise(N);
	
	% 2. Generative Model
	sigma_mu = exp(mu) ./ sum(exp(mu),1);
	g_x = (p_x * sigma_mu);
	g_y = (p_y * sigma_mu);
	
	% 3. Prediction Error
	epsilon_x = s_x - g_x;
	epsilon_y = s_y - g_y;
	
	% 4. Update Parameters
	d_px = k_p * epsilon_x * sigma_mu';
	d_py = k_p * epsilon_y * sigma_mu';
	p_x = Integrate(p_x, d_px);
	p_y = Integrate(p_y, d_py);
	
	% Simple convergence check
	delta = sum(abs(d_px) + abs(d_py) , 'all');
	if delta < 1e-5
		fprintf("Convergence! Delta : %f\n", delta)
		break;
	end
	
	% Calculate Variational Free Energy
	vfe = (epsilon_x/2).^2 + (epsilon_y/2).^2;
	vfe_sum = sum(vfe,'all');
	
	
	
	%% Plot
	try
		% Draw on intervals only
		if mod(t,drawInt) ~= 0
			continue
		end
		
		% Prints these variables to console
		Debug( "PARAM X", p_x, "PARAM Y", p_y, ...
			"VFE", vfe(:,idx), "Total VFE", vfe_sum)
		
		% Update Cell Scatter Plot
		SetAll(hmain, {'XData','YData','CData'}, ...
			{psi_x(1,:),psi_x(2,:),sigma_mu'})
		titletext = sprintf("N: %d | k_p: %.2f | dt: %.2f | Time: %.2f", ...
				N, k_p, dt, dt*t);
		ht.String = titletext;
		
		drawnow
	catch ME
		warning("Drawing loop broken. Error given: '%s'", ME.message)
		break
	end
end

fprintf("Training Ended. Final Parameters:\nP_X:\n")
disp(p_x)
fprintf("P_Y:\n")
disp(p_y)



%% Helper Functions

function mu = BayerFilter(N)
% Bayer Filter (works only with odd square roots of N)
	idx = reshape(mod(1:N,2), [sqrt(N),sqrt(N)]);
	idx = idx .* repmat( mod(1:sqrt(N),2)*2+1 , sqrt(N), 1);
	idx(idx==0) = 2;
	mu = zeros(3,N);
	for i = 1:N
		mu(idx(i),i) = 1;
	end
end

function pos = SquareLattice(N, d)
% Generate positions in a square lattice, where each cell is atleast a distance
% of `d` apart (by the edge). Requires square number of cells, `N`.
	s = sqrt(N);
	range = 0 : d : d*s-d;
	[X,Y] = meshgrid(range);
	
	center = mean(range);
	X = X - center;
	Y = Y - center;
	pos = [X(:),Y(:)]';
end

function omega = Noise(N)
% Noise Generation Function
	omega = sqrt(1/exp(16)) * randn([3,N]);
end

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

function Debug(varargin)
% Prints variables to console
% Input String + Variable pair
% e.g.: Debug("Variable Name", var_name, ...)

	if length(varargin) > 1
		fprintf("\n=========\n")
	end
	
	for i = 1:2:length(varargin)
		fprintf(varargin{i} + "\n")
		disp(varargin{i+1})
	end
end