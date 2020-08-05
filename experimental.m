%% Simulation parameters
clear
clc

GIF = false;
filename = 'output.gif';

% Drawing interval
drawInt = 1;

% Axis display range
axRange = 5;
% Hard boundary?
boundary = true;
% Axis Lock?
axLock = false;
% Heatmap Grid Spacing
hmSpace = 0.2;


% Cell types
C = 3;

% Number of cells
% Nr = 27; % Red
% Ng = 18; % Green
% Nb = 9; % Blue
% Nc = [Nb Ng Nr];
% N = sum(Nc);

% N = C * 20;

Nc = [1 2 3];
N = sum(Nc);

% Time step size
dt = 0.05;
tLimit = 1000;

% Anonymous dt Update function
Integrate = @(x,dx) x + (dt.*dx);

%% Cell properties

% ======================== Prior ========================= %
% Secretion Prior
priorSec = eye(C);

% Position Prior
% ones in a line of width 3 across the diagonal
priorPos = ones(C);
priorPos( tril(priorPos,-2) | triu(priorPos,2) ) = 0; 

% priorPos = rand(C);

% priorPos = [1 1 0 0
% 			1 1 1 0
% 			0 1 1 1
% 			0 0 1 1];

% Inference Prior
priorInfer = zeros(3);

% ======================== Inference ========================= %
% infer = randn(C,N);

infer = [];
for i = 1:C
	new = repmat(priorSec(:,i), 1, Nc(i));
	infer = [infer, new];
end

infer = softmax(infer);

% ======================== Position ========================= %
% pos = randn(2,N);

pos = [];
for i = 1:C
	new = [cos(0: 2*pi/(Nc(i)) :2*pi) ; sin(0: 2*pi/(Nc(i)) :2*pi)] * (i)/2;
	new(:,end) = [];
	pos = [pos, new];
end

% ======================== Secretion ========================= %
sec = zeros(C,N);

% ======================== Variance ========================= %
% Secretion Sensor Variance
varSec = ones(C,N);
% Position Sensor Variance
varPos = ones(C,N);
% Inference Variance
varInfer = ones(C,N);

% ======================== Sensor ======================== %
% Secretion Sensor
senseSec = zeros(C,N);
% Position Sensor
sensePos = zeros(C,N);

% ==================== Sensor Error ==================== %
% Secretion Sensor Error
errSec = zeros(C,N);
% Position Sensor Error
errPos = zeros(C,N);

% ==================== Inference Error ==================== %
% Inference Error
errInfer = zeros(C,N);


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
[~,ctype] = max(infer);

% Scatter Plot
ax1 = subplot(2,2,1);
hmain = scatter(pos(1,:), pos(2,:), 100, ctype, 'filled', ...
	'MarkerEdgeColor', 'flat');
ht = title(sprintf("N: %d | dt: %.3f | Ready. Press key to begin.", N, dt));
ax1.CLim = [1 C]; % Allow for many colors
grid on
hold on
hquiv = quiver(pos(1,:), pos(2,:), zeros(1,N), zeros(1,N), 'k');

% Heatmap Mu 1
ax2 = subplot(2,2,2);
hphi1 = pcolor(X,Y,hmap1);
title("\mu_1")
grid on

% Heatmap Mu 2
ax3 = subplot(2,2,3);
hphi2 = pcolor(X,Y,hmap2);
title("\mu_2")
grid on

% Heatmap Mu 3
ax4 = subplot(2,2,4);
hphi3 = pcolor(X,Y,hmap3);
title("\mu_3")
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
SetAll([ax2,ax3,ax4], 'CLim', [0 1])
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
	% ------------------- Sensor ------------------- %
	% 1. Sensory Inputs
	senseSec = sec + Noise(C,N);
	sensePos = DistSensor(pos, sec, N) + Noise(C,N);
	
	% 2. Softmax Function
	% Selected cell type based on secretion
	belief = softmax(infer);
	[~,ctype] = max(belief);
	
	% ------------------- Error ------------------- %
	% 3.1 Perception Error
	errSec = (senseSec - (priorSec(:,ctype) .* belief));
	errPos = (sensePos - (priorPos(:,ctype) .* belief));
	% 3.2 Inference Error
	errInfer = (infer - priorInfer(:,ctype));
	
	% ------------------- Variance ------------------- %
	% 4.1 Perception Variance
	varSec = errSec.^2;
	varPos = errPos.^2;
	% 4.2 Inference Variance
	varInfer = errInfer.^2;
	
	Debug("Sec", varSec, "Pos", varPos, "Infer", varInfer)
	
	% ------------------- Update ------------------- %
	% 5.1 Update Secretion
	dSec = -errSec .* pinv(varSec)';
	sec = Integrate(sec, dSec);
	
	% 5.2 Update Position
	dPos = SensorGrad(pos, sec, C, N);
	dPos = PositionUpdate(eye(C), dPos, errPos, N);
	pos = Integrate(pos, dPos);
	
	% Boundary Condition
	if boundary
		pos( pos < -axRange ) = -axRange;
		pos( pos > axRange ) = axRange;
	end
	
	% 5.3 Update Inference
	invGen = InverseGenModel(priorSec, infer, C, N);
	dInferSec = InferenceUpdate(...
		invGen, errSec, errPos, pos, priorSec(:,ctype), infer, C, N);
	infer = Integrate(infer, dInferSec);
	
% 	Debug("Inference", infer(:,1), "Secretion", sec(:,1), "Position", pos(:,1))
	
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
		% Update cell center to be axis center
		if axLock
			% Position change relative to overall cluster movement
			dPos = dPos - mean(dPos,2);
			pos = pos - mean(pos,2);
		end
		
		% Update Cell Scatter
		SetAll(hmain, {'XData','YData','CData'}, {pos(1,:),pos(2,:),ctype})
		ht.String = sprintf("N: %d | dt: %.2f | Time: %.2f", N, dt, t*dt);
		
		% Update Heatmaps
		SetAll([hphi1;hphi2;hphi3], {'CData'}, {hmap1;hmap2;hmap3});
		
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

function omega = Noise(C, N)
% Noise Generation Function
% Noise with precision of exp(16)
	omega = exprnd(sqrt(1/exp(16)), [C,N]);
end

function S = DistSensor (pos, sec, N)
% Distance Sensor function
% Calculate the extracellular input for each cell
% Assuming distance function is squared Euclidean distance
% Input: 
%	[2,N]	: position
% 	[C,N]	: secretion
% 	scalar	: N
% Output: 
% 	[C,N]	: sensor

	% Spatial decay constant -- See DEM.m
	k = 2;
	d = pdist(pos', 'squaredeuclidean');
	S = sec * (exp(-k * squareform(d)) - eye(N));
end

function gradient = SensorGrad (pos, sec, C, N)
% Sensory Gradient function
% Calculate the sensory gradient for each cell
% Input: 
%	[2,N]	: position
% 	[C,N]	: secretion
% 	scalar	: C
% 	scalar	: N
% Output: 
% 	[2,C,N]	: gradient

	% [X,Y] are [j,i] matrices
	X = repmat(pos(1,:), [N,1]);
	Y = repmat(pos(2,:), [N,1]);
	
	% Spatial decay constant -- See DEM.m
	k = 2;
	
	% Pairwise Exponential Distance Decay Matrix
	% (From normal S_Alpha calculations)
	dd = pdist(pos', 'squaredeuclidean');
	dd = exp(-k * squareform(dd)) - eye(N);
	
	% Partial Derivatives w.r.t. x/y
	% Becareful of the shape of X,Y; the transpose order matters
	dx = X - X'; % (x_j - x_i) | [N,N]
	dy = Y - Y'; % (y_j - y_i) | [N,N]
	
	% Calculate Partial Derivative
	% [3,N] = -2 .* k .* ([C,N] * ([N,N] .* [N,N]))
	dSdx = -2 .* k .* sec * (dx .* dd); 
	dSdy = -2 .* k .* sec * (dy .* dd);
	
	% Gradient matrix, [2,C,N]
	gradient = zeros(2,C,N);
	for i = 1:N
		gradient(1,:,i) = dSdx(:,i)';
		gradient(2,:,i) = dSdy(:,i)';
	end
end

function dPos = PositionUpdate(prec, grad, error, N)
% Position Update function
% Input: 
%	[C,C]	: precision
% 	[2,C,N] : gradient
% 	[3,N]	: error
% Output: 
% 	[3,N]	: dPos
	dPos = zeros(2,N);
	
	% Iterate through each cell
	% Essentially, each cell gets its own [2,C] "gradient matrix", which
	% modulates how the sensory error is "perceived" in each direction, thus
	% modulating how the cell moves.
	for i = 1:N
		% [2,1]   = -([2,C,1]	  * [C,C]) * [C,1]
		dPos(:,i) = -(grad(:,:,i) * prec) * error(:,i);
	end
end

function invGen = InverseGenModel(prior, infer, C, N)
% Inverse of the generative model
% Function of the derivative of softmax function
% Input:  
%	[C,C]	: prior
%	[C,N]	: inference
% Output: 
%	[C,N]	: inverse

% 	% Assuming d_sigma is [C,N]
% 	invGen = zeros(C,N);
% 	for i = 1:N
% 		% [C,N] = ( [C,C] - [C,1]*[1,C] ) * [C,1]
% 		invGen(:,i) = (diag(mu(:,i)) - (mu(:,i)*mu(:,i)')) * prior(:,i);
% 	end
	
	% Assuming d_sigma is [C,C,N]
	invGen = zeros(C,C,N);
	for i = 1:N
		% [C,C,N] = [C,C] * ( [C,C] - [C,1]*[1,C] )
		invGen(:,:,i) = prior * (diag(infer(:,i)) - (infer(:,i)*infer(:,i)'));
	end
end

function dInfer = InferenceUpdate(invGen, errSec, errPos, pos, prior, infer, C, N)
% Inference Update function
% Calculates the change to inference
% Input: 
% 	[C,N]	: invGen
% 	[C,N]	: errSec
% 	[C,N]	: errPos
% 	[C,N]	: pos
%	[C,N]	: prior
%	[C,N]	: infer
% 	scalar	: N
% Output: 
% 	[C,N]	: dInfer

	% Distance Decay matrix [N,N]
	k = 2;
	d = pdist(pos', 'squaredeuclidean');
	% Position Error decay [C,N]
	errPosDecay = errPos * (exp(-k * squareform(d)) - eye(N));
	
	% Inference Error [C,N]
	errInfer = infer - prior;
	
% 	% Assuming inverse of GenModel is [C,N]
% 	dInfer = invGen .* (errSec + errPosDecay) - errInfer;
	
	% Assuming inverse of GenModel is [C,C,N]
	dInfer = zeros(C,N);
	for i = 1:N
		dInfer(:,i) = invGen(:,:,i) * (errSec(:,i) + errPosDecay(:,i)) ...
			- errInfer(:,i);
	end
end

function sig_maps = Heatmap (X, Y, pos, sec)
% Heatmap Calculations
% Input:
%	[P,P]	: X | x-coordinates of gradient vector arrows, [P,P] matrix
%	[P,P]	: Y | y-coordinates of gradient vector arrows, [P,P] matrix
%	[2,N]	: psi_x | cell coordinates
%	[C,N]	: psi_y | cell chemical signals
% Output:
%	{[P*P,1], [P*P,1], [P*P,1]} : sig_maps | signal heatmap for each signal type
%		(Up to 3 types shown)

	% Distance from each cell to each reference point in each dimensions
	x_diff = repmat(pos(1,:), numel(X), 1) - X(:);
	y_diff = repmat(pos(2,:), numel(Y), 1) - Y(:);
	
	% Distance function (exponential decay of squared euclidean distance)
	k = 2;
	euc_dist = x_diff.^2 + y_diff.^2;
	dist_decay = exp(-k .* euc_dist);
	
	% Decay of each signal across grid -- Signal Heatmap (Type 1-3)
	mu1 = sum(dist_decay.*sec(1,:),2);
	mu2 = sum(dist_decay.*sec(2,:),2);
	mu3 = sum(dist_decay.*sec(3,:),2);
	
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