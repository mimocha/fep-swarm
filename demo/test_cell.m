%% Signal Heatmap Demo
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020
%
% This file tests the trained cell-like position in its trained state. (see
% dissertation section 3.2.1)
% Test around with the initial parameters to see different results.
%
% Used in conjunction with `train_cell.m`



%% Simulation parameters
clear
clc

% Save GIF video?
GIF = false;
filename = "output.gif";
% Drawing interval
drawInt = 10;
% Viewport display range
axRange = 3;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.2;
% Draw cell movement vectors?
showMoves = false;

% Number of cells / Starting States
Nr = 9;	% Red Cells
Ng = 6;	% Green Cells
Nb = 1;	% Blue Cells
N = Nr + Ng + Nb;	% Total Number of Cells

% Time step size
dt = 0.01;
% Time Limit
tLimit = 200;
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
p_x =  [
   -0.4771    0.8801    1.2905
   -1.5822    1.3211    3.7720
   -0.7059    1.1955    1.1954
];
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
% mu = mu + randn(3,N)/4;

% =============== Belief ===================================================== %
sigma_mu = exp(mu) ./ sum(exp(mu),1);

% =============== Cell Position ============================================== %
% Random Initial Positions
% psi_x = rand(2,N) * axRange - axRange/2;

% Cell-like Initial Position
x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 1.5;
x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 0.5;
x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0;
psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
% Add noise to initial position
% psi_x = psi_x + randn(2,N)*0.2;

% =============== Cell Signals =============================================== %
% Initialize with signal emitted
psi_y = softmax(mu);

% Initialize without signal
% psi_y = zeros(3,N);

% =============== Sensor States ============================================== %
s_x = zeros(3,N); % Extracellular
s_y = zeros(3,N); % Intracellular

% =============== Active States ============================================== %
a_x = zeros(2,N); % Extracellular
a_y = zeros(3,N); % Intracellular

% =============== Prediction Error =========================================== %
epsilon_x = zeros(3,N); % Extracellular
epsilon_y = zeros(3,N); % Intracellular



%% Heatmap Mesh Grid
R = -axRange:hmSpace:axRange;
[X,Y] = meshgrid(R, R);

% Heatmap functions
HMShape = @(hmap, X) reshape(hmap,size(X));
sig_maps = Heatmap (X, Y, psi_x, psi_y);
hmap1 = HMShape(sig_maps{1},X);
hmap2 = HMShape(sig_maps{2},X);
hmap3 = HMShape(sig_maps{3},X);



%% Figure setup
figure(1)
clf
colormap jet
cmap = sigma_mu'; % Color cells based on belief

% Scatter Plot
ax1 = subplot(2,2,1);
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
		
		% Signal Mapping Function
		sig_maps = Heatmap (X, Y, psi_x, psi_y);
		hmap1 = HMShape(sig_maps{1},X);
		hmap2 = HMShape(sig_maps{2},X);
		hmap3 = HMShape(sig_maps{3},X);
		
		% Update Heatmaps
		SetAll([hmu1;hmu2;hmu3], {'CData'}, {hmap1;hmap2;hmap3});
		
		drawnow
		
		% Save GIF
		if GIF
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		end
	catch ME
		warning("Drawing loop broken. Error given: '%s'", ME.message)
		break
	end
end




%% Functions

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

function sig_maps = Heatmap (X, Y, psi_x, psi_y)
% Heatmap Calculations
% Input:
%	[P,P]	: X : x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y : y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x : cell positions
%	[3,N]	: psi_y : cell signals
% Output:
%	{[P*P,1], [P*P,1], [P*P,1]} : sig_maps | signal heatmap for each signal type

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

function SaveGIF (h, filename, mode, mode2)
% GIF Exporting Function

	frame = getframe(h);
	im = frame2im(frame);
	[imind,cm] = rgb2ind(im,256);
	imwrite(imind,cm,filename,mode,mode2,'DelayTime',0);
end