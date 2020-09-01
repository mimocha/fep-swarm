function a_x = PositionUpdate (grad, error, N)
% Position Update function
% Input: 
% 	[2,3,N] : grad  : sensory gradient
% 	[3,N]	: error : extracellular error
% Output: 
% 	[3,N]	: a_x : Active state (cell movement)

	a_x = zeros(2,N);
	% For each cell
	for i = 1:N
		% [2,1]  = -[2,3,1] * [3,1]
		a_x(:,i) = -grad(:,:,i) * error(:,i);
	end
end