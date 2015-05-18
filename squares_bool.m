function [Ui] = squares_bool(x, n)
   xx = true(size(x, 1), size(x, 2) - (n-1));
   xy = true(size(x, 1) - (n-1), size(x, 2) - (n-1));
   for j = 1:n
	   xx = xx & x(:, j:j+size(xx,2)-1);
   end
   for j = 1:n
	   xy = xy & xx(j:j+size(xy,2)-1, :);
   end
   Ui = xy;
   Ui = [Ui, false(size(Ui, 1), n - 1)];
   Ui = [Ui; false(n - 1, size(Ui, 2))];
end
