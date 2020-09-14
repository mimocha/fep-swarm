function Debug(varargin)
% Debug variable console print
% Input String + Variable pair
% e.g.: Debug("Variable Name", var_name, ...)

	if length(varargin) > 1
		fprintf("\n=========\n")
	end
	
	for i = 1:2:length(varargin)
		fprintf(varargin{i} + "\n")
		disp(varargin{i+1})
	end
end