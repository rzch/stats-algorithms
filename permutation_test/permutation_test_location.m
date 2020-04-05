clc
clear
close all

% This script demonstrates a permutation test for a location parameter

%%

%sample size
n = 10;

%set random seed
rng(42);

%normal sample
X = randn(n, 1);

%found for confidence interval
m0 = -0.525;
m0 = 0.7;

%null hypothesis
m0 = -0.9;

%sample deviations from null
Xd = X - m0;

%test statistic
d = sum(Xd);

S = zeros(2^n, n);
%binary representation of signs, code segment adapted from 'de2bi' function
for i = 1:2^n                  % Cycle through each element of the input vector/matrix.
    j = 1;
    tmp = i - 1;
    while (j <= n) && (tmp > 0)     % Cycle through each digit.
        S(i, j) = rem(tmp, 2);      % Determine current digit.
        tmp = floor(tmp/2);
        j = j + 1;
    end
end
S = fliplr(S);
S(S == 0) = -1;

%permutation distribution
T = zeros(2^n, 1);
for i = 1:2^n
   T(i) = sum(S(i, :).*Xd');
end
histogram(T)
sigma = std(T);

%one-sided p-value (alternative greater than)
p = sum(T > d)/(2^n);
%two-sided p-value
p2 = (sum(T > abs(d)) + sum(T < -abs(d)))/(2^n);