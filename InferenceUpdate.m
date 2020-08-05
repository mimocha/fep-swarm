function dInfer = InferenceUpdate(priorSec, priorPos, errSec, errPos, S, C, N)
% Inference Update function
% Calculates the change to inference
% Input: 
% 	[C,C]	: priorPos
% 	[C,C]	: priorPos
% 	[C,N]	: errSec
% 	[C,N]	: errPos
% 	[C,N]	: S -- Softmax belief
% 	scalar	: C -- Cell type
% 	scalar	: N -- Cell number
% Output: 
% 	[C,N]	: dInfer
	
	% Inverse softmax
	invSoftmax = zeros(C,C,N);
	for i = 1:N
		invSoftmax(:,:,i) = (diag(S(:,i)) - (S(:,i)*S(:,i)'));
	end

	% d_mu = -(Px + Py) * sigma'(mu) * (eps_x + eps_y)
	errSum = errSec + errPos;
	priorSum = priorSec + priorPos;
	dInfer = zeros(C,N);
	for i = 1:N
		dInfer(:,i) = - priorSum * invSoftmax(:,:,i) * errSum(:,i);
	end
end