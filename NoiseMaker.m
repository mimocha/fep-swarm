function omega = NoiseMaker(N)
	omega = exprnd(sqrt(1/exp(16)), [3,N]);
end