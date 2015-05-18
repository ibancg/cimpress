function [Ui] = squares_conv(x, i)
   C = conv2(x, ones(i)) == i^2;
   Ui = C(i:end, i:end);
end
