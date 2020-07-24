%% Simulation parameters
clear
clc

GIF = true;
filename = 'Heatmap_07.gif';

% Drawing interval
drawInt = 10;

% Axis display range
axRange = 5;
% Axis Lock?
axLock = true;
% Heatmap Grid Spacing
hmSpace = 0.1;

% Number of cells
N = 30;

% Time step size
dt = 0.001;
tLimit = 30;

%% Cell properties
% Generate N random cells

% Intracellular belief prior
prior_y = eye(3);
% Extracellular belief prior
prior_a =  [1 1 0; ...
			1 1 1; ...
			0 1 1];
% prior_a = prior_a * 0.1;
% prior_a = randn(3);

% ======================== [3,1] vector | Beliefs ========================= %
% Diverges to infinity if sum of any column is greater than 1
% mu = softmax(rand(3,N));

% mu = [repmat([1;0;0],1,9), repmat([0;1;0],1,6), [0;0;1]];
% mu = softmax(mu);

mu = zeros(3,N);
for i = 1:N
	mu(randi(3), i) = 1;
end
		
% ====================== [2,1] vector | Cell position ====================== %
psi_x = randn(2,N);

% psi_x = [ cos(0:2*pi/N:2*pi) ; sin(0:2*pi/N:2*pi) ];
% psi_x(:,end) = [];

% x1 = [cos(0:2*pi/9:2*pi) ; sin(0:2*pi/9:2*pi)] * 2;
% x2 = [cos(0:2*pi/6:2*pi) ; sin(0:2*pi/6:2*pi)] * 1;
% x3 = [0 ; 0];
% psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3];

% ==================== [3,1] vector | Cell secretion ==================== %
% psi_y = randn(3,N); % Important to use RANDN and not RAND
% psi_y = softmax(mu);
psi_y = zeros(3,N);

% ==================== Sensor ==================== %
% [3,1] vector | Intracellular sensor
s_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular sensor
s_a = zeros(3,N); % Init values doesn't matter

% ==================== Prediction Error ==================== %
% [3,1] vector | Intracellular prediction error
epsilon_y = zeros(3,N); % Init values doesn't matter
% [3,1] vector | Extracellular prediction error
epsilon_a = zeros(3,N); % Init values doesn't matter


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
colormap jet
cmap = mu';

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready. Press key to begin.", N, dt));
grid on

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
	epsilon_y = s_y - (prior_y * sigma_mu);
	epsilon_a = s_a - (prior_a * sigma_mu);
	
	% 4.1 Update Secretion
	da_y = -epsilon_y;
	psi_y = psi_y + (dt .* da_y);
	
	% 4.2 Update Position
	grad_S = DeriveAlpha(psi_x, s_a, N);
	da_x = DxUpdate(eye(3), grad_S, epsilon_a, N);
	psi_x = psi_x + (dt .* da_x);
	
	% 5. Update Beliefs
	d_sigma = DeriveSoftmax(mu, N);
	d_mu = MuUpdate(eye(3), d_sigma, epsilon_a, N);
	mu = mu + (dt .* d_mu);
	
	
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
		% Update Cell Scatter
		SetAll(hmain, {'XData','YData','CData'}, {psi_x(1,:),psi_x(2,:),mu'})
		ht.String = sprintf("N: %d | dt: %.3f | Time: %.3f", N, dt, t*dt);
		
		% Update Heatmaps
		SetAll([hmu1;hmu2;hmu3], {'CData'}, {hmap1;hmap2;hmap3});

		% Update axis limit for visibility
		if axLock
			psi_x = psi_x - mean(psi_x,2);
		end
		
		drawnow
		
		if GIF
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		end
	catch
		fprintf("Drawing loop broken\n")
		break
	end
	
% 	Debug("S_Y", s_y, "PSI_Y", psi_y, "EPSILON_Y", epsilon_y)
end



%% Functions

% Distance function
function s_a = Alpha (psi_x, psi_y, N)
	% Default Euclidean distance assumption
	% Returns [3,N] matrix
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	if N > 1
		d = pdist(psi_x', 'squaredeuclidean');
		s_a = psi_y * (exp(-k * squareform(d)) - eye(N));
	else
		s_a = zeros(3,1);
	end
end

