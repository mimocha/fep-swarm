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