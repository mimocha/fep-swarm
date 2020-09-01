	%% Simulation Setup
% Chris' code with Palacios' notations

clear;
% rng(6);

% simulation parameters
tLimit = 100;
dt = 0.005;
time =  0:dt:tLimit;
N = length(time);
action = true;

% Generative Model Parameters
prior = 4; % desired temperature

% The time that action onsets
actionTime = tLimit / 4;

% sensory variances
Omega_z0 = 0.1;
Omega_z1 = 0.1;

% hidden state variances
Omega_w0 = 0.1;
Omega_w1 = 0.1;

% Params for generative process
T0 = 100; % temperature at x = 0

% Initialise brain state variables
% Beliefs
mu_0 = nan(1,N);
mu_0(1) = 0;
mu_1 = nan(1,N);
mu_1(1) = 0;
mu_2 = nan(1,N);
mu_2(1) = 0;

% Sensory noise in the generative process
zgp_0 = randn(1,N) * 0.1;
zgp_1 = randn(1,N) * 0.1;

% Actions
a = nan(1,N);
a(1) = 0;

% Initialise generative process (World)
da_x = nan(1,N);
da_x(1) = a(1);
psi_x = nan(1,N);
psi_x(1) = 2;

psi_y = nan(1,N);
psi_y(1) = T0 / (psi_x(1)^2 + 1);
grad_psi_y = nan(1,N);
grad_psi_y(1) = -2 * T0 * psi_x(1) * (psi_x(1)^2 + 1)^(-2);
psi_y_dt = nan(1,N);
psi_y_dt(1) = grad_psi_y(1) * (da_x(1));

% Initialise sensory input
s_0 = nan(1,N);
s_0(1) = psi_y(1);

s_1 = nan(1,N);
s_1(1) = psi_y_dt(1);

% Initialise error terms
epsilon_z_0 = (s_0(1) - mu_0(1));
epsilon_z_1 = (s_1(1) - mu_1(1));
epsilon_w_0 = (mu_1(1) + mu_0(1) - prior);
epsilon_w_1 = (mu_2(1) + mu_1(1));

% Initialise Variational Energy
VFE = nan(1,N);
VFE(1) =  1/Omega_z0 * epsilon_z_0^2/2 ...
		+ 1/Omega_z1 * epsilon_z_1^2/2 ...
		+ 1/Omega_w0 * epsilon_w_0^2/2 ...
		+ 1/Omega_w1 * epsilon_w_1^2/2 ...
		+ 1/2 * log(Omega_w0 * Omega_w1 * Omega_z0 * Omega_z1);

% Gradient descent learning parameters
k = 0.1; % for inference
ka = 0.01; % for learning

%% Simulation
for t = 2:N
	%% The generative process (i.e. the real world)
	% Position
	da_x(t) = a(t-1); % action
	psi_x(t) = psi_x(t-1) + dt * da_x(t); % Integrate position
	
	% TODO: This needs to be changed to psi_y functions
	% Temperature
	psi_y(t) = T0 / (psi_x(t)^2 + 1);
	% Temperature Gradient
	grad_psi_y(t) = -2 * T0 * psi_x(t) / (psi_x(t)^2 + 1)^2;
	% Derivative of temperature w.r.t. time
	psi_y_dt(t) = grad_psi_y(t) * (da_x(t));
	
	% Sense Current Temperature
	s_0(t) = psi_y(t) + zgp_0(t);
	% Sense Rate of Temperature Change
	s_1(t) = psi_y_dt(t) + zgp_1(t);

	%% The generative model (i.e. the agents brain)
	% Perception Error
	epsilon_z_0 = (s_0(t-1) - mu_0(t-1));
	epsilon_z_1 = (s_1(t-1) - mu_1(t-1));
	% Inference Error
	epsilon_w_0 = (mu_1(t-1) + mu_0(t-1) - prior);
	epsilon_w_1 = (mu_2(t-1) + mu_1(t-1));
	
	% Free Energy
	VFE(t) =  1 / Omega_z0 * (epsilon_z_0^2) / 2 ...
			+ 1 / Omega_z1 * (epsilon_z_1^2) / 2 ...
			+ 1 / Omega_w0 * (epsilon_w_0^2) / 2 ...
			+ 1 / Omega_w1 * (epsilon_w_1^2) / 2 ...
			+ 1 / 2 * log(Omega_w0 * Omega_w1 * Omega_z0 * Omega_z1);
	
	% Belief
	mu_0(t) = mu_0(t-1) ...
			+ dt * ( mu_1(t-1) - k ...
			* (-epsilon_z_0/Omega_z0 ...
			  + epsilon_w_0/Omega_w0) );
	mu_1(t) = mu_1(t-1) ...
			+ dt * ( mu_2(t-1) - k ...
			* (-epsilon_z_1/Omega_z1 ...
			  + epsilon_w_0/Omega_w0 ...
			  + epsilon_w_1/Omega_w1) );
	mu_2(t) = mu_2(t-1) ...
			+ dt * -k * (epsilon_w_1/Omega_w1);

	% Allow action after set time
	if (time(t) > actionTime)
		% active inference
		a(t) = a(t-1) ...
			+ dt * -ka * grad_psi_y(t) * epsilon_z_1 / Omega_z1;
	else
		a(t) = 0;
	end
end

%% Plot

figure(1);
clf;

subplot(5,1,1)
hold on
plot(time,psi_y);
plot(time,psi_x);
legend('psi_y','psi_x')

subplot(5,1,2)
hold on
plot(time,mu_0,'k');
plot(time,mu_1,'m');
plot(time,mu_2,'b');
legend("\mu'","\mu''","\mu'''");

subplot(5,1,3)
hold on
plot(time,s_0,'k');
plot(time,s_1,'m');
legend("s","s'");

subplot(5,1,4)
hold on
plot(time,a,'k');
ylabel('a')

subplot(5,1,5)
plot(time,VFE,'k');
xlabel('time');
ylabel('VFE')



