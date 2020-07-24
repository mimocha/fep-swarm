% Cell class definition
classdef Cell
	properties (Constant)
		% Intracellular belief prior
		prior_y = eye(3);
		
		% Extracellular belief prior
		prior_a = [1 1 0; 1 1 1; 0 1 1];
	end
	
	properties
		psi_x % [2,1] vector | Cell position
		psi_y % [3,1] vector | Cell secretion
		
		s_y % [3,1] vector | Intracellular sensor
		s_a % [3,1] vector | Extracellular sensor
		
		mu % [3,1] vector | Beliefs
		
		epsilon_y % [3,1] vector | Intracellular prediction error
		epsilon_a % [3,1] vector | Extracellular prediction error
	end
	
	methods
		% Constructor
		function self = Cell ()
			self.psi_x = zeros(2,1);
			self.psi_y = zeros(3,1);
			self.s_y = zeros(3,1);
			self.s_a = zeros(3,1);
		end
		
		function self = CellSignal (self, others)
			% Iteratively accumulates signal from other cells
			% This could be vectorized
			si = zeros(3,1);
			for i = 1:numel(others)
				si = si + SignalDecay(self, others(i));
			end
			
			% No noise for now
			% Noise of low variance sigma == sqrt(1/exp(16))
			self.s_a = si;
		end
		
		function s = SignalDecay (self, other)
			% signal value for each of the other cells decay exponentially based
			% on distance between cells. (Palacios, eq.10) Assuming the equation
			% to be euclidean norm of the distance between cells, and not a
			% simple absolute function.
			s = exp(- sum((self.psi_x - other.psi_x).^2) );
			
			% Use distance to attenuate signals of other cells.
			s = s .* other.psi_y;
		end
		
		function self = GenerativeModel (self)
			
		end
	end
end