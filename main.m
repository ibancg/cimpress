x = [
              [0,0,1,1,1,1,1,1,1,0,0,1],
              [1,0,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [0,0,0,0,1,1,1,1,1,1,1,0],
              [0,0,0,0,1,1,1,1,1,1,1,0],
              [0,0,0,0,1,1,1,1,1,1,0,0],
              [0,0,0,0,1,1,1,1,1,1,1,1],
              ];
x = (x == 1);

% x = rand(40, 100);
% x = (x > 0.2);

M = size(x,1)
N = size(x,2)

n = min(size(x))
U = {};
Uxy = {};
m = [];

for i = 1:n
   Ui = squares_bool(x, i);
   mi = sum(sum(Ui));
   if (mi == 0)
      break;
   end
   m(i) = mi;
   U{i} = Ui;
end

n = length(m)

[i, l] = place(x, n)

x_ = int16(x);
for j = 1:length(i)
	[ix_, iy_] = ind2sub(size(x), i(j));
	n = l(j);
	x_(ix_:ix_+(n-1), iy_:iy_+(n-1)) = j + 1;
end

figure(1)
subplot(121);
imagesc(x); axis equal;
subplot(122);
imagesc(x_); axis equal;
