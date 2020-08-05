function omega = Noise(C, N)
% Noise Generation Function
	omega = randn(sqrt(1/exp(16)), [C,N]);
end