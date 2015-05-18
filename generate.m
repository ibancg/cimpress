function [U] = generate(m, n, nSquares, maxSize)

	U = false(m, n);
	maxSize = min(maxSize, min(m, n));
	for i = 1:nSquares
		size = 	1 + floor(maxSize*rand());
		x = 1 + floor((n - size + 1)*rand());
		y = 1 + floor((m - size + 1)*rand());
		U(x:x+size-1, y:y+size-1) = true(size, size);
	end

end