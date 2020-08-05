%% Simulation parameters
clear
clc

GIF = false;
filename = 'new-gradient-03.gif';

% Drawing interval
drawInt = 10;
% Axis display range
axRange = 4;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.2;

% Number of cells
Nr = 16; % Red
Ng = 8; % Green
Nb = 1; % Blue
N = Nr + Ng + Nb;
% N = 18;

% Time step size
dt = 0.05;
tLimit = 1000;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);


%% Cell properties
% =================== Generative Parameter ==================== %
% Secretion Parameter
p_y =  [1 0 0; ...
			0 1 0; ...
			0 0 1];
% Position Parameter
p_x =  [1 1 0; ...
			1 1 1; ...
			0 1 1];

% ======================== Inference ========================= %
% mu = zeros(3,N);
% mu = randn(3,N);

mu = [	repmat([1;0;0],1,Nr) , ...
		repmat([0;1;0],1,Ng) , ...
		repmat([0;0;1],1,Nb) ];

% mu = mu / 2 + randn(3,N)/8;
	
sigma_mu = exp(mu) ./ sum(exp(mu),1);

% ======================== Position ========================= %
% psi_x = rand(2,N) * axRange - axRange/2;

% psi_x = [ cos(0:2*pi/N:2*pi) ; sin(0:2*pi/N:2*pi) ] * 3;
% psi_x(:,end) = [];

try
	x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 2;
	x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 1;
	x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0.5;
	psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
catch
	Nr = floor(N/2);
	Ng = floor(N/3);
	Nb = N - Nr - Ng;
	
	x1 = [cos(0: 2*pi/Nr :2*pi) ; sin(0: 2*pi/Nr :2*pi)] * 2.00;
	x2 = [cos(0: 2*pi/Ng :2*pi) ; sin(0: 2*pi/Ng :2*pi)] * 1.25;
	x3 = [cos(0: 2*pi/Nb :2*pi) ; sin(0: 2*pi/Nb :2*pi)] * 0.50;
	psi_x = [x1(:,1:end-1), x2(:,1:end-1), x3(:,1:end-1)];
end
% psi_x = psi_x + randn(2,N)*0.3;

% ======================== Secretion ========================= %
% psi_y = zeros(3,N);
psi_y = softmax(mu);

% ======================== Sensor ======================== %
% Secretion Sensor
s_y = zeros(3,N);
% Position Sensor
s_x = zeros(3,N);

% ==================== Sensor Error ==================== %
% Secretion Sensor Error
epsilon_y = zeros(3,N);
% Position Sensor Error
epsilon_x = zeros(3,N);


idx = [1, 2, floor(N/2), floor(N/2)+1, N-1, N];

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
figure('Position', [680 90 875 888], 'Units', 'pixels')
clf
colormap jet
cmap = sigma_mu';

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(psi_x(1,:), psi_x(2,:), 100, cmap, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready", N, dt));
grid on
hold on
xticks(-axRange:axRange)
yticks(-axRange:axRange)
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

MU = nan(3, length(idx), length(1:tLimit/dt));
DMU = nan(3, length(idx), length(1:tLimit/dt));

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
	s_x = DistSensor(psi_x, psi_y, N);
	
	% 2. Generative Model
	sigma_mu = exp(mu) ./ sum(exp(mu),1);
	g_y = (p_y * sigma_mu);
	g_x = (p_x * sigma_mu);
	
	% 3. Perception Error
	epsilon_y = s_y - g_y;
	epsilon_x = s_x - g_x;
	
	% 4.1 Update Secretion
	da_y = -epsilon_y;
	psi_y = Integrate(psi_y, da_y);
	
	% 4.2 Update Position
	grad_s = SensorGrad(psi_x, psi_y, 3, N);
	da_x = PositionUpdate(eye(3), grad_s, epsilon_x, N);
	psi_x = Integrate(psi_x, da_x);
	
	% Boundary Condition
	if boundary
		psi_x( psi_x < -axRange ) = -axRange;
		psi_x( psi_x > axRange ) = axRange;
	end
	
	% 5. Update Beliefs
	d_mu = InferenceUpdate(p_y, p_x, epsilon_y, epsilon_x, sigma_mu, 3, N);
	d_mu = 0.1 .* d_mu;
	mu = Integrate(mu, d_mu);
	
	% Tracking Belief
	MU(:,:,t) = mu(:,idx);
	DMU(:,:,t) = d_mu(:,idx);
	
	% 6. Variational Free Energy
	VFE = sum(epsilon_x.^2,1)/2 + sum(epsilon_y.^2,1)/2;
	
	% Draw on intervals only
	if mod(t,drawInt) ~= 0
		continue
	end
	
	Debug("VFE", VFE(:,idx), ...
		  "MU", mu(:,idx), ...
		  "PSI_Y == SIGMA_MU", psi_y(:,idx), ...
		  "EPSILON_Y", epsilon_y(:,idx), ...
		  "EPSILON_A", epsilon_x(:,idx), ...
		  "D_MU", d_mu(:,idx))
	
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
		SetAll(hmain, {'XData','YData','CData'}, ...
			{psi_x(1,:),psi_x(2,:),sigma_mu'})
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

return

%% Plot Belief

T = 1:tLimit/dt;

figure
subplot(2,1,1)
hold on
for i = 1:length(idx)
	plot(T, squeeze(MU(1,i,:)), 'r', ...
		 T, squeeze(MU(2,i,:)), 'g', ...
		 T, squeeze(MU(3,i,:)), 'b');
end
title("MU")

subplot(2,1,2)
hold on
for i = 1:length(idx)
	plot(T, squeeze(DMU(1,i,:)), 'r', ...
		 T, squeeze(DMU(2,i,:)), 'g', ...
		 T, squeeze(DMU(3,i,:)), 'b');
end
title("D MU")




