clc
clear
close all

% This script demonstrates the coverage probability of a confidence 
% interval for a location parameter by inverting a permutation test

%%

%sample size
n = 10;

%level of significance
alpha = 0.05;

%actual location parameter
theta = 0;

%set random seed
rng(42);

%simulate confidence intervals for coverage probability
Nsim = 100;
count = 0;
for N = 1:Nsim
   
    [lb, ub] = conf_int(n, alpha, theta);
    
    if (lb <= theta && theta <= ub)
        count = count + 1;
    end
end

coverage_prob = count/Nsim;

function [lb, ub] = conf_int(n, alpha, theta)

%normal sample
X = theta + randn(n, 1);

m0s = linspace(-2, 2, 200);

for k = 1:length(m0s)
    %null hypothesis
    m0 = m0s(k);
    
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
    
    %one-sided p-value (alternative greater than)
    p(k) = sum(T > d)/(2^n);
    %two-sided p-value
    p2(k) = (sum(T > abs(d)) + sum(T < -abs(d)))/(2^n);
    
end

%confidence interval
lb = min(m0s(p2 >= alpha));
ub = max(m0s(p2 >= alpha));

end


