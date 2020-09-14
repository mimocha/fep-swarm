function omega = Noise(N)
% Noise Generation Function
%
% For the MSc Dissertation:
% A Free Energy Principle approach to modelling swarm behaviors
% Chawit Leosrisook, MSc Intelligent and Adaptive Systems
% School of Engineering and Informatics, University of Sussex, 2020

	omega = sqrt(1/exp(16)) * randn([3,N]);
end