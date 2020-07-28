%% Simulation parameters
clear
clc

GIF = false;
filename = 'output.gif';

% Drawing interval
drawInt = 50;

% Axis display range
axRange = 5;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.1;

% Number of cells
Nr = 9; % Red
Ng = 6; % Green
Nb = 3; % Blue
N = Nr + Ng + Nb;
% N = 14;

% Time step size
dt = 0.01;
tLimit = 100;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);

%% Cell properties
% ======================== Prior ========================= %
% Secretion Prior
priorSec =  [1 0 0; ...
			 0 1 0; ...
			 0 0 1];
% Position Prior
priorPos =  [1 1 0; ...
			 1 1 1; ...
			 0 1 1];

% ======================== Inference ========================= %
% Secretion Inference
inferSec = [repmat([1;0;0],1,Nr) , repmat([0;1;0],1,Ng) , repmat([0;0;1],1,Nb)];
inferSec = ArgMax(inferSec);
% Position Inference
inferPos = [repmat([1;1;0],1,Nr) , repmat([1;1;1],1,Ng) , repmat([0;1;1],1,Nb)];
inferPos = ArgMax(inferPos);

% ======================== Position ========================= %
pos = randn(2,N);

% pos = [ cos(0:2*pi/N:2*pi) ; sin(0:2*pi/N:2*pi) ];
% pos(:,end) = [];

% x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 2.00;
% x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 1.25;
% x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0.50;
% pos = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];

% pos = [-0.5, 0.5; 0, 0];

% ======================== Secretion ========================= %
% sec = zeros(3,N);
sec = ArgMax(inferSec);

% ======================== Sensor ======================== %
% Secretion Sensor
senseSec = zeros(3,N);
% Position Sensor
sensePos = zeros(3,N);

% ==================== Sensor Error ==================== %
% Secretion Sensor Error
errSenSec = zeros(3,N);
% Position Sensor Error
errSenPos = zeros(3,N);

% ==================== Inference Error ==================== %
% Secretion Inference Error
errInfSec = zeros(3,N);
% Position Inference Error
errInfPos = zeros(3,N);


%% Heatmap Mesh Grid
R = -axRange:hmSpace:axRange;
[X,Y] = meshgrid(R, R);

% Heatmap function
HMShape = @(hmap, X) reshape(hmap,size(X));
sig_maps = Heatmap (X, Y, pos, sec);
hmap1 = HMShape(sig_maps{1},X);
hmap2 = HMShape(sig_maps{2},X);
hmap3 = HMShape(sig_maps{3},X);


%% Figure setup
figure(1)
clf
colormap jet
cmap = inferSec';

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(pos(1,:), pos(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready. Press key to begin.", N, dt));
grid on
hold on
hquiv = quiver(pos(1,:), pos(2,:), zeros(1,N), zeros(1,N), 'k');

% Heatmap Mu 1
ax2 = subplot(2,2,2);
hphi1 = pcolor(X,Y,hmap1);
title("\mu_1 (Red)")
grid on

% Heatmap Mu 2
ax3 = subplot(2,2,3);
hphi2 = pcolor(X,Y,hmap2);
title("\mu_2 (Green)")
grid on

% Heatmap Mu 3
ax4 = subplot(2,2,4);
hphi3 = pcolor(X,Y,hmap3);
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
SetAll([hphi1,hphi2,hphi3], 'EdgeColor', 'None')


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
	senseSec = sec;
	sensePos = SensorFunc(pos, sec, N);
	
	% 2. Generative Model
	[~,idx] = max(inferSec);
	genSec = (priorSec(:,idx) .* ArgMax(inferSec));
	[~,idx] = max(inferPos);
	genPos = (priorPos(:,idx) .* ArgMax(inferPos));
	
	% 3.1 Perception Error
	errSenSec = senseSec - genSec;
	errSenPos = sensePos - genPos;
	
% 	% 3.2 Prediction Error ----- TODO: Prior?
% 	[~,idx] = max(inferSec);
% 	errInfSec = inferSec - priorSec(:,idx);
% 	[~,idx] = max(inferPos);
% 	errInfPos = inferPos - priorPos(:,idx);
	
% 	e_mu_phi = mu_phi - g_phi;
% 	e_mu_chi = mu_chi - g_chi;
	
	% 4.1 Update Secretion
	dSec = -errSenSec;
	sec = Integrate(sec, dSec);
	
	% 4.2 Update Position
	dSenPos = DeriveSensorFunc(pos, sec, N);
	dPos = DxUpdate(eye(3), dSenPos, errSenPos, N);
	pos = Integrate(pos, dPos);
	
	% Boundary Condition
	if boundary
		pos( pos < -axRange ) = -axRange;
		pos( pos > axRange ) = axRange;
	end
	
% 	% 5.1 Update Secretion Inference
% 	dGenPhi = DeriveGenModel(priorSec, inferSec, N);
% 	dInferSec = MuUpdate(dGenPhi, errSenSec, N);
% 	inferSec = Integrate(inferSec, dInferSec);
% 	
% 	% 5.2 Update Position Inference
% 	dGenChi = DeriveGenModel(priorPos, inferPos, N);
% 	dInferPos = MuUpdate(dGenChi, errSenPos, N);
% 	inferPos = Integrate(inferPos, dInferPos);
	
% 	Debug("Position Error", errInfPos, "Secretion Error", errInfSec)
	
	% Draw
	if mod(t,drawInt) ~= 0
		continue
	end
	
	
	%% Signal Mapping Function
	sig_maps = Heatmap (X, Y, pos, sec);
	hmap1 = HMShape(sig_maps{1},X);
	hmap2 = HMShape(sig_maps{2},X);
	hmap3 = HMShape(sig_maps{3},X);
	
	%% Plot
	try
		% Update Cell Scatter
		SetAll(hmain, {'XData','YData','CData'}, {pos(1,:),pos(2,:),inferSec'})
		ht.String = sprintf("N: %d | dt: %.3f | Time: %.3f", N, dt, t*dt);
		
		% Update Heatmaps
		SetAll([hphi1;hphi2;hphi3], {'CData'}, {hmap1;hmap2;hmap3});
		
		% Update cell center to be axis center
		if axLock
			pos = pos - mean(pos,2);
			
			% Position change relative to overall cluster movement
			dPos = dPos - mean(dPos,2);
		end
		
		% Update Cell Quiver
		SetAll(hquiv, {'XData','YData','UData','VData'}, ...
			{pos(1,:),pos(2,:),dPos(1,:),dPos(2,:)})
		
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

% Max function
function x = ArgMax (x)
% 	x = x ./ sum(x,1);
	x = softmax(x);
end

% Distance function
% Calculate the sensory input for each cell
% Assuming distance function is squared Euclidean distance
% Input: 
%	[2,N]	: psi_x
% 	[3,N]	: psi_y
% 	scalar	: N
% Output: 
% 	[3,N]	: s_a
function s_a = SensorFunc (chi, phi, N)
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	if N > 1
		d = pdist(chi', 'squaredeuclidean');
		s_a = phi * (exp(-k * squareform(d)) - eye(N));
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
function grad_S = DeriveSensorFunc (psi_x, psi_y, N)
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
function d_sigma = DeriveGenModel(prior, infer, N)
	% Derivative of softmax
	d_sigma = zeros(3,3,N);
	for i = 1:N
		% [3,3,1]	   = [3,3]		   - ([3,1]  *  [1,3])
		d_sigma(:,:,i) = diag(infer(:,i)) - (infer(:,i)*infer(:,i)');
		
		% Times prior (dot product)
		d_sigma(:,:,i) = prior * d_sigma(:,:,i);
	end
end

% Mu Update Calculation function
% Calculates the change to mu
% Input: 
% 	[3,3,N] : d_sigma
% 	[3,N]	: epsilon_s
% 	[3,N]	: epsilon_mu
%	scalar	: N
% Output: 
% 	[3,N]	: d_mu
function d_mu = MuUpdate(d_sigma, epsilon_s, N)
	d_mu = zeros(3,N);
	
	% Iterate through each cell
	for i = 1:N
		% [3,1]   = - [3,3,1] * [3,1]
		d_mu(:,i) = - (d_sigma(:,:,i)) * epsilon_s(:,i);
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