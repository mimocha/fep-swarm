%% Gradient Quiver Demo
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020
%
% This demo shows the movement gradient, based on each signal type, as
% overlapping vector arrows. These colored arrows show which direction each
% signal is "pulling" each cell type to.
%
% Although it's not entirely accurate, as the vector calculations are done on
% top of the existing cells. In other words, this is just another (less useful)
% way of drawing the chemical gradient.



%% Simulation parameters
clear
clc

% Save GIF video?
GIF = false;
filename = "output.gif";
% Drawing interval
drawInt = 20;
% Viewport display range
axRange = 3;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Gradient Quiver Spacing
gqSpace = 0.5;
% Draw cell movement vectors?
showMoves = true;

% Number of cells / Starting States
Nr = 9;	% Red Cells
Ng = 6;	% Green Cells
Nb = 1;	% Blue Cells
N = Nr + Ng + Nb;	% Total Number of Cells

% Time step size
dt = 0.01;
% Time Limit
tLimit = 100;
% Start Time
t = 0;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);



%% Cell properties
% =============== Learning Rates ============================================= %
k_a = 1; % Action
k_mu = 1; % Belief

% =============== Generative Parameter ======================================= %
% Extracellular Parameters
p_x =  [1 1 0; ...
		1 1 1; ...
		0 1 1];
% Intracellular Parameters
p_y =  [1 0 0; ...
		0 1 0; ...
		0 0 1];

% =============== Internal States ============================================ %
% Initial internal states
mu = [	repmat([1;0;0],1,Nr) , ...
		repmat([0;1;0],1,Ng) , ...
		repmat([0;0;1],1,Nb) ];
% Add noise to initial internal states
mu = mu + randn(3,N)/4;

% =============== Belief ===================================================== %
sigma_mu = exp(mu) ./ sum(exp(mu),1);

% =============== Cell Position ============================================== %
% Random Initial Positions
% psi_x = rand(2,N) * axRange - axRange/2;

% Cell-like Initial Position
x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 1;
x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 0.5;
x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0;
psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
% Add noise to initial position
psi_x = psi_x + randn(2,N)*0.2;

% =============== Cell Signals =============================================== %
% Initialize with signal emitted
% psi_y = softmax(mu);

% Initialize without signal
psi_y = zeros(3,N);

% =============== Sensor States ============================================== %
s_x = zeros(3,N); % Extracellular
s_y = zeros(3,N); % Intracellular

% =============== Active States ============================================== %
a_x = zeros(2,N); % Extracellular
a_y = zeros(3,N); % Intracellular

% =============== Prediction Error =========================================== %
epsilon_x = zeros(3,N); % Extracellular
epsilon_y = zeros(3,N); % Intracellular



%% Quiver Plot Mesh Grid
R = -axRange:gqSpace:axRange;
[X,Y] = meshgrid(R, R);

% Generative model g(mu_i)
G = {repmat(reshape(p_x*softmax([1,0,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(p_x*softmax([0,1,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(p_x*softmax([0,0,1]'), [1,1,3]), size(X))};

[U,V] = GradientMap (G, X, Y, psi_x, psi_y);



%% Figure setup
figure(1)
clf
cmap = sigma_mu'; % Color cells based on belief

% Scatter Plot
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
titletext = sprintf("N: %d | k_a: %.2f | k_\\mu: %.2f | dt: %.2f | Time: %.2f", ...
				N, k_a, k_mu, dt, dt*t);
ht = title(titletext);
grid on
hold on
xticks(-axRange:axRange)
yticks(-axRange:axRange)
if showMoves
	hquiv = quiver(psi_x(1,:), psi_x(2,:), zeros(1,N), zeros(1,N), 'k');
end

% Quiver Plot
hquiv1 = quiver(X,Y,U{1},V{1},'r');
hquiv2 = quiver(X,Y,U{2},V{2},'g');
hquiv3 = quiver(X,Y,U{3},V{3},'b');

% Axis Tracking variables
axT = axRange;
axB = -axRange;
axL = -axRange;
axR = axRange;

% Styling
SetAll = @(H, propName, propVal) set(H, propName, propVal);
grid on
daspect([1 1 1])
axis([axL axR axB axT])



%% Simulation loop

fprintf("Ready. Press any key to begin ...\n")
pause

if GIF
	fig = gcf;
	SaveGIF(fig, filename, 'LoopCount', inf);
end

for t = 1:tLimit/dt
	% 1. Sensory Inputs
	s_x = DistSensor(psi_x, psi_y, N) + Noise(N);
	s_y = psi_y + Noise(N);
	
	% 2. Generative Model
	sigma_mu = exp(mu) ./ sum(exp(mu),1); % softmax
	g_x = (p_x * sigma_mu);
	g_y = (p_y * sigma_mu);
	
	% 3. Prediction Error
	epsilon_x = s_x - g_x;
	epsilon_y = s_y - g_y;
	
	% 4. Calculate Action
	grad = SensorGrad(psi_x, psi_y, N);
	a_x = -k_a * PositionUpdate(grad, epsilon_x, N);
	a_y = -k_a * epsilon_y;
	
	% 5. Update World
	psi_x = Integrate(psi_x, a_x);
	psi_y = Integrate(psi_y, a_y);
	
	% World Boundary Condition
	if boundary
		psi_x( psi_x < -axRange ) = -axRange;
		psi_x( psi_x > axRange ) = axRange;
	end
	
	% 6. Update Beliefs
	d_mu = InternalStatesUpdate(p_y, p_x, epsilon_y, epsilon_x, sigma_mu, N);
	d_mu = k_mu * d_mu;
	mu = Integrate(mu, d_mu);
	
	
	
	%% Plot
	try
		% Draw on intervals only
		if mod(t,drawInt) ~= 0
			continue
		end
		
		% Lock ensemble center to axis center
		if axLock
			% Position change relative to overall cluster movement
			psi_x = psi_x - mean(psi_x,2);
			a_x = a_x - mean(a_x,2);
		end
		
		% Update Cell Scatter Plot
		SetAll(hmain, {'XData','YData','CData'}, ...
			{psi_x(1,:),psi_x(2,:),sigma_mu'})
		titletext = sprintf("N: %d | k_a: %.2f | k_\\mu: %.2f | dt: %.2f | Time: %.2f", ...
			N, k_a, k_mu, dt, t*dt);
		ht.String = titletext;
		
		% Update Cell Quiver Arrows
		if showMoves
		SetAll(hquiv, {'XData','YData','UData','VData'}, ...
			{psi_x(1,:),psi_x(2,:),a_x(1,:),a_x(2,:)})
		end
		
		% Gradient Mapping Function
		[U,V] = GradientMap (G, X, Y, psi_x, psi_y);
		SetAll(hquiv1, {'UData','VData'}, {U{1},V{1}})
		SetAll(hquiv2, {'UData','VData'}, {U{2},V{2}})
		SetAll(hquiv3, {'UData','VData'}, {U{3},V{3}})
		
		drawnow
		
		if GIF
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		end
	catch ME
		warning("Drawing loop broken. Error given: '%s'", ME.message)
		break
	end
end



%% Helper Functions

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

function grad = SensorGrad (pos, sig, N)
% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: pos : cell position
% 	[3,N]	: sig : cell signal
% 	scalar	: N   : cell count
% Output: 
% 	[2,3,N]	: gradient

	% [X,Y] are [j,i] matrices
	X = repmat(pos(1,:), [N,1]);
	Y = repmat(pos(2,:), [N,1]);
	
	% Pairwise Exponential Distance Decay Matrix
	k = 2; % Spatial decay constant -- See DEM.m from spm12 toolkit
	dd = pdist(pos', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([3,N] * ([N,N] .* [N,N]))
	dsdx = -2 .* k .* sig * (dx .* dd); 
	dsdy = -2 .* k .* sig * (dy .* dd);
	
	% Gradient matrix, [2,3,N]
	grad = zeros(2,3,N);
	for i = 1:N
		grad(1,:,i) = dsdx(:,i)';
		grad(2,:,i) = dsdy(:,i)';
	end
end

function a_x = PositionUpdate (grad, error, N)
% Position Update function
% Input: 
% 	[2,3,N] : grad  : sensory gradient
% 	[3,N]	: error : extracellular error
% Output: 
% 	[3,N]	: a_x : Active state (cell movement)

	a_x = zeros(2,N);
	% For each cell
	for i = 1:N
		% [2,1]  = [2,3,1] * [3,1]
		a_x(:,i) = grad(:,:,i) * error(:,i);
	end
end

function d_mu = InternalStatesUpdate (p_x, p_y, eps_x, eps_y, s_mu, N)
% Internal States Update function
% Calculates the change to internal states
% Input: 
% 	[3,3]	: p_x   : extracellular parameters
% 	[3,3]	: p_y   : intracellular parameters
% 	[3,N]	: eps_x : prediction error
% 	[3,N]	: eps_y : prediction error
% 	[3,N]	: s_mu  : belief
% 	scalar	: N     : cell count
% Output: 
% 	[3,N]	: d_mu  : internal states update
	
	% d_mu = k_mu * sigma'(mu) * (p_x * eps_x + p_y * eps_y)
	d_mu = zeros(3,N);
	for i = 1:N
		invSoftmax = (diag(s_mu(:,i)) - (s_mu(:,i)*s_mu(:,i)'));
		d_mu(:,i) = invSoftmax * (p_x * eps_x(:,i) + p_y * eps_y(:,i));
	end
end

function [U, V] = GradientMap (G, X, Y, psi_x, psi_y)
% Gradient Quiver Calculations
% Input:
%	{[P,P,3], [P,P,3], [P,P,3]}	: G (fixed generative model g(mu))
%	[P,P]	: X : x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y : y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x : cell positions
%	[3,N]	: psi_y : cell signals
% Output:
%	{[P,P,3], [P,P,3], [P,P,3]} : U : x-component of gradient vector
%	{[P,P,3], [P,P,3], [P,P,3]} : V : y-component of gradient vector

	% Distance from each cell to each reference point in each dimensions
	% X_j - X_i
	dx = X(:) - repmat(psi_x(1,:), numel(X), 1);
	dy = Y(:) - repmat(psi_x(2,:), numel(Y), 1);
	
	% Pairwise Exponential Distance Decay
	k = 2; % Spatial decay constant -- See DEM.m
	dd = dx.^2 + dy.^2;
	dd = exp(-k .* dd);
	
	% ------------------- SENSOR ERROR ------------------- %
	
	% Decay of each signal across grid (Signal Heatmap)
	% [P^2,1] = [P^2,N] * [N,1]
	psi1_decay = dd * psi_y(1,:)';
	psi2_decay = dd * psi_y(2,:)';
	psi3_decay = dd * psi_y(3,:)';
	% Merge into [P,P,3] matrix
	psi_decay = cat(3, psi1_decay, psi2_decay, psi3_decay);
	psi_decay = reshape(psi_decay, [size(X),3]);
	
	% Epsilon Alpha Calculations
	epsilon_mu1 = psi_decay - G{1}; % [P,P,3]
	epsilon_mu2 = psi_decay - G{2}; % [P,P,3]
	epsilon_mu3 = psi_decay - G{3}; % [P,P,3]
	
	% ------------------- GRADIENT ------------------- %
	
	% Calculate Partial Derivative (Signal Gradient)
	% PSI 1
	psi1_dSdx = -2 .* k .* psi_y(1,:) .* (dx .* dd);
	psi1_dSdy = -2 .* k .* psi_y(1,:) .* (dy .* dd);
	% PSI 2
	psi2_dSdx = -2 .* k .* psi_y(2,:) .* (dx .* dd);
	psi2_dSdy = -2 .* k .* psi_y(2,:) .* (dy .* dd);
	% PSI 3
	psi3_dSdx = -2 .* k .* psi_y(3,:) .* (dx .* dd);
	psi3_dSdy = -2 .* k .* psi_y(3,:) .* (dy .* dd);
	
	% Sum element contributions, reshape to grid (grad_S equivalent)
	% PSI 1
	psi1_dSdx = reshape(sum(psi1_dSdx,2), size(X));
	psi1_dSdy = reshape(sum(psi1_dSdy,2), size(Y));
	% PSI 2
	psi2_dSdx = reshape(sum(psi2_dSdx,2), size(X));
	psi2_dSdy = reshape(sum(psi2_dSdy,2), size(Y));
	% PSI 3
	psi3_dSdx = reshape(sum(psi3_dSdx,2), size(X));
	psi3_dSdy = reshape(sum(psi3_dSdy,2), size(Y));
	
	% Combine into propagation matrix
	grid_dx = cat(3, psi1_dSdx, psi2_dSdx, psi3_dSdx);
	grid_dy = cat(3, psi1_dSdy, psi2_dSdy, psi3_dSdy);
	
	% ------------------- DX ------------------- %
	
	% Change to dot_A_x
	% MU_A = -dSxy .* epsilon
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

function SaveGIF (h, filename, mode, mode2)
% GIF Exporting Function

	frame = getframe(h);
	im = frame2im(frame);
	[imind,cm] = rgb2ind(im,256);
	imwrite(imind,cm,filename,mode,mode2,'DelayTime',0);
end