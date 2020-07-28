%% Simulation parameters
clear
clc

% Save GIF?
GIF = true;
filename = 'output.gif';
% Drawing interval
drawInt = 20;
% Axis display range
axRange = 5;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.5;

% Number of cells / Starting States
% Comment these 3 out, and set N directly to have randomly set starting mu
Nr = 27; % Red
Ng = 18; % Green
Nb = 9; % Blue
N = Nr + Ng + Nb;
% N = 50;

% Time step size
dt = 0.05;
tLimit = 250;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);


%% Cell properties
% ======================== Prior ========================= %
% Secretion Prior
prior_y =  [1 0 0; ...
			0 1 0; ...
			0 0 1];
% Position Prior
prior_a =  [1 1 0; ...
			1 1 1; ...
			0 1 1];

% ======================== Inference ========================= %
try
	mu = [repmat([1;0;0],1,Nr) , repmat([0;1;0],1,Ng) , repmat([0;0;1],1,Nb)];
catch
	mu = randn(3,N);
end
mu = softmax(mu);

% ======================== Position ========================= %
psi_x = rand(2,N) * axRange - axRange/2;

% x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 2.00;
% x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 1.00;
% x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0.50;
% psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
% psi_x = psi_x + randn(2,N)*0.5;

% ======================== Secretion ========================= %
psi_y = zeros(3,N);

% ======================== Sensor ======================== %
% Secretion Sensor
s_y = zeros(3,N);
% Position Sensor
s_a = zeros(3,N);

% ==================== Sensor Error ==================== %
% Secretion Sensor Error
epsilon_y = zeros(3,N);
% Position Sensor Error
epsilon_a = zeros(3,N);


%% Quiver Plot Mesh Grid
R = -axRange:hmSpace:axRange;
[X,Y] = meshgrid(R, R);

% Generative model g(mu_i)
G = {repmat(reshape(prior_a*softmax([1,0,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(prior_a*softmax([0,1,0]'), [1,1,3]), size(X)), ...
	 repmat(reshape(prior_a*softmax([0,0,1]'), [1,1,3]), size(X))};

[U,V] = GradientMap (G, X, Y, psi_x, psi_y);


%% Figure setup
figure(1)
clf
cmap = mu';

% Scatter Plot
h = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.2f | Ready", N, dt));
grid on
hold on
xticks(-axRange:axRange)
yticks(-axRange:axRange)
hquiv = quiver(psi_x(1,:), psi_x(2,:), zeros(1,N), zeros(1,N), 'k');

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

fprintf("Ready.\n")
pause

% GIF
if GIF
	fig = gcf;
	SaveGIF(fig, filename, 'LoopCount', inf);
end

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
	[~,idx] = max(sigma_mu);
	epsilon_y = s_y - (prior_y(:,idx) .* sigma_mu);
	epsilon_a = s_a - (prior_a(:,idx) .* sigma_mu);
	
	% 4.1 Update Secretion
	da_y = -epsilon_y;
	psi_y = Integrate(psi_y, da_y);
	
	% 4.2 Update Position
	grad_S = DeriveAlpha(psi_x, psi_y, N);
	da_x = DxUpdate(eye(3), grad_S, epsilon_a, N);
	psi_x = Integrate(psi_x, da_x);
	
	% Boundary Condition
	if boundary
		psi_x( psi_x < -axRange ) = -axRange;
		psi_x( psi_x > axRange ) = axRange;
	end
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	d_mu = MuUpdate(eye(3), d_sigma, epsilon_y, N);
	mu = Integrate(mu, d_mu);
	
	% Draw on intervals only
	if mod(t,drawInt) ~= 0
		continue
	end
	
	%% Gradient Mapping Function
	[U,V] = GradientMap (G, X, Y, psi_x, psi_y);
	
	%% Plot
	try
		% Update Cell Scatter
		SetAll(h, {'XData','YData','CData'}, {psi_x(1,:),psi_x(2,:),mu'})
		ht.String = sprintf("N: %d | dt: %.2f | Time: %.2f", N, dt, t*dt);
		
		% Update Quiver Plots
		SetAll([hquiv1;hquiv2;hquiv3], {'UData','VData'}, {U{1},V{1};U{2},V{2};U{3},V{3}})
		
		% Update cell center to be axis center
		if axLock
			% Position change relative to overall cluster movement
			psi_x = psi_x - mean(psi_x,2);
			da_x = da_x - mean(da_x,2);
		end
		
		% Update Cell Quiver
		SetAll(hquiv, {'XData','YData','UData','VData'}, ...
			{psi_x(1,:),psi_x(2,:),da_x(1,:),da_x(2,:)})
		
		drawnow
		
		if GIF
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		end
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
% 	[3,N]	: psi_y
% 	scalar	: N
% Output: 
% 	[2,3,N]	: grad_S
function grad_S = DeriveAlpha (psi_x, psi_y, N)
	% [X,Y] are [j,i] matrices
	X = repmat(psi_x(1,:), [N,1]);
	Y = repmat(psi_x(2,:), [N,1]);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Pairwise Exponential Distance Decay Matrix
	% (From normal S_Alpha calculations)
	dd = pdist(psi_x', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([3,N] * ([N,N] .* [N,N]))
	dSdx = -2 .* k .* psi_y * (dx .* dd); 
	dSdy = -2 .* k .* psi_y * (dy .* dd);
	
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
	% X_j - X_i
	dx = X(:) - repmat(psi_x(1,:), numel(X), 1);
	dy = Y(:) - repmat(psi_x(2,:), numel(Y), 1);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Pairwise Exponential Distance Decay
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

% GIF Exporting Function
function SaveGIF (h, filename, mode, mode2)
	frame = getframe(h);
	im = frame2im(frame);
	[imind,cm] = rgb2ind(im,256);
	imwrite(imind,cm,filename,mode,mode2,'DelayTime',0);
end