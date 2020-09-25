%% Main Simulation File
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020
%
% This is the main "test bench" file for playing around with whatever.
% Helper functions are saved as separate files.



%% Simulation parameters
clear
clc

% Save GIF video?
GIF = false;
filename = "./result/output";

% Drawing interval
drawInt = 20;
% Axis display range
axRange = 3;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.2;
% Movement quiver
showMoves = false;

% Number of cells
Nr = 9; % Red
Ng = 6; % Green
Nb = 1; % Blue
N = Nr + Ng + Nb;

% Time step size
dt = 0.01;
% Time limit
tLimit = 50;
% Time
t = 0;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);

% Anonymous softmax function
Softmax = @(x) exp(x)./sum(exp(x),1);



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
mu = mu / 2 + randn(3,N)/4;

% =============== Belief ===================================================== %
sigma_mu = Softmax(mu);

% =============== Cell Position ============================================== %
% Random Initial Positions
% psi_x = rand(2,N) * axRange - axRange/2;

% Cell-like Initial Position
x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 1.5;
x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 0.5;
x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0;
psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
% Add noise to initial position
psi_x = psi_x + randn(2,N)*0.25;

% =============== Cell Signals =============================================== %
% psi_y = Softmax(mu);
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
figure('Position', [680 90 875 888], 'Units', 'pixels')
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

% Axis OuterPosition
ax1.OuterPosition = [0, 0.5, 0.5, 0.5];
ax2.OuterPosition = [0.5, 0.5, 0.5, 0.5];
ax3.OuterPosition = [0, 0, 0.5, 0.5];
ax4.OuterPosition = [0.5, 0, 0.5, 0.5];

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
	sigma_mu = Softmax(mu);
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
			[~,blueidx] = max(sigma_mu);
			blueidx = blueidx == 3;
			% Position change relative to overall cluster movement
			psi_x = psi_x - mean(psi_x(:,blueidx),2);
			a_x = a_x - mean(a_x(:,blueidx),2);
		end
		
		% Update Cell Scatter Plot
		SetAll(hmain, {'XData','YData','CData'}, ...
			{psi_x(1,:),psi_x(2,:),sigma_mu'})
		titletext = sprintf("N: %d | k_a: %.2f | k_\\mu: %.2f | dt: %.2f | Time: %.2f", ...
				N, k_a, k_mu, dt, dt*t);
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
		
		if GIF
			SaveGIF(fig, filename, 'WriteMode', 'Append');
		end
	catch ME
		warning("Drawing loop broken. Error given: '%s'", ME.message)
		break
	end
end

disp("Simulation Done!")
