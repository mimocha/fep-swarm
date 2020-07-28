%% Simulation parameters
clear
clc

GIF = false;
filename = 'new-gradient-03.gif';

% Drawing interval
drawInt = 20;
% Axis display range
axRange = 5;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.1;

% Number of cells
% Nr = 27; % Red
% Ng = 18; % Green
% Nb = 9; % Blue
% N = Nr + Ng + Nb;
N = 100;

% Time step size
dt = 0.05;
tLimit = 1000;

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

% psi_x = [ cos(0:2*pi/N:2*pi) ; sin(0:2*pi/N:2*pi) ] * 3;
% psi_x(:,end) = [];

% x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 2.00;
% x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 1.00;
% x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0.50;
% psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
% psi_x = psi_x + randn(2,N)*0.2;

% ======================== Secretion ========================= %
psi_y = zeros(3,N);
% psi_y = softmax(mu);

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


%% Heatmap Mesh Grid
R = -axRange:hmSpace:axRange;
[X,Y] = meshgrid(R, R);

% Heatmap function
HMShape = @(hmap, X) reshape(hmap,size(X));
sig_maps = Heatmap (X, Y, psi_x, psi_y);
hmap1 = HMShape(sig_maps{1},X);
hmap2 = HMShape(sig_maps{2},X);
hmap3 = HMShape(sig_maps{3},X);


%% Figure setup
figure(1)
clf
colormap jet
cmap = mu';

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready", N, dt));
grid on
hold on
hquiv = quiver(psi_x(1,:), psi_x(2,:), zeros(1,N), zeros(1,N), 'k');

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
	
	%% Signal Mapping Function
	sig_maps = Heatmap (X, Y, psi_x, psi_y);
	hmap1 = HMShape(sig_maps{1},X);
	hmap2 = HMShape(sig_maps{2},X);
	hmap3 = HMShape(sig_maps{3},X);
	
	%% Plot
	try
		% Update cell center to be axis center
		if axLock
			% Position change relative to overall cluster movement
			psi_x = psi_x - mean(psi_x,2);
			da_x = da_x - mean(da_x,2);
		end
		
		% Update Cell Scatter
		SetAll(hmain, {'XData','YData','CData'}, {psi_x(1,:),psi_x(2,:),mu'})
		ht.String = sprintf("N: %d | dt: %.2f | Time: %.2f", N, dt, t*dt);
		
		% Update Heatmaps
		SetAll([hmu1;hmu2;hmu3], {'CData'}, {hmap1;hmap2;hmap3});
		
		% Update Cell Quiver
		SetAll(hquiv, {'XData','YData','UData','VData'}, ...
			{psi_x(1,:),psi_x(2,:),da_x(1,:),da_x(2,:)})
		
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

