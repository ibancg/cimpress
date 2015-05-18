% place squares of size n and less
function [i,l,v] = place(x, n)

	fprintf(1, ">> n = %i\n" , n)
	fflush(1);

	if (n == 1)
		i = find(x);
		i = i';
		l = ones(size(i));
		v = [];
	else
	
		U = squares_bool(x, n);
		i__ = find(U)'
		
		if (~isempty(i__))
			nBest = Inf;

			% sets of squares we can place
			sets = {};
			scores = [];
			
			for j = 1:length(i__)
				i_ = i__(j);
				x_ = x;
				[ix_, iy_] = ind2sub(size(x), i_);
				x_(ix_:ix_+(n-1), iy_:iy_+(n-1)) = false;
				[i___, l___, v___] = place(x_, n);
				n_ = length(l___);
				if (n_ < nBest)
					i = [i_, i___];
					l = [n, l___];
%					break;
				end
			end
		else
			[i, l] = place(x, n - 1);
		    keyboard;
		end	

	end
end