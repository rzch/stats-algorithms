% This script implements a basic subspace identfication algorithm on a toy
% system

clc
clear
close all

%%

%set random number seed
rng(42)

%define system matrices
A = [0.7 -0.01; 0.5 0.01];
%A = [0.9 -0.01; 0.01 0.9]; %alternative
B = [0.5 -0.3; 0.4 0.1];
C = [3 4; -2 -1];
D = [1 1; -1 0];
%D = [0 0; 0 0]; %alternative

%system dimensions
n = size(A, 1);
m = size(B, 2);
p = size(C, 1);

%noise covariance matrices
E = 0.01*eye(n);
F = 0.01*eye(p);

%cholesky decomposition of noise covariance matrices
sqrtE = chol(E);
sqrtF = chol(F);

%inital condition
x0 = [1; 1];
%x0 = [0; 0]; %alternative

%number of samples to simulate for
Nsignal = 2000;
%white noise input signal
Usignal = 2*randn(m, Nsignal); 
%first input
u0 = 2*randn(m, 1);

Ysignal = zeros(p, Nsignal);
Xsim = zeros(n, Nsignal);

xcurr = x0;
uprev = u0;

%simulate dynamics with noise
for k = 1:Nsignal
    Xsim(:, k) = A*xcurr + B*uprev + sqrtE*randn(n, 1);
    Ysignal(:, k) = C*Xsim(:, k) + D*Usignal(:, k) + sqrtF*randn(p, 1);
    
    uprev = Usignal(:, k);
    xcurr = Xsim(:, k);
    
end

%maximal prediction horizon
r = 3;
N = Nsignal - r + 1;

Y = zeros(r*p, N);
U = zeros(r*m, N);
X = zeros(n, N);

%construct block matrices
for k = 1:N
    Yrk = reshape(Ysignal(:, k:(k + r - 1)), r*p, 1);
    Urk = reshape(Usignal(:, k:(k + r - 1)), r*m, 1);
    Y(:, k) = Yrk;
    U(:, k) = Urk;
    X(:, k) = Xsim(:, k);
end

%projection matrix
M = eye(N) - U'*((U*U')\U);

%backward horizons
s1 = 2;
s2 = 2;
s = p*s1 + m*s2;
Phi1 = zeros(s1*p, N);
Phi2 = zeros(s2*m, N);

%construct instrumental variables matrices
Phi1(1:p, :) = [zeros(p, 1), Ysignal(:, 1:(N - 1))];
for i = 1:(s1 - 1)
    Phi1(i*p + (1:p), (i + 1):N) = Phi1(1:p, 1:(N - i));
end

Phi2(1:m, :) = [u0, Usignal(:, 1:(N - 1))];
for i = 1:(s2 - 1)
    Phi2(i*m + (1:m), (i + 1):N) = Phi2(1:m, 1:(N - i));
end

Phi = [Phi1; Phi2];

%estimate extended observability matrix
G = Y*M*Phi'/N;
%G = Y*M/N; %alternative
[Ucal, Sigma, Vcal] = svd(G);
Ucal1 = Ucal(1:r*p, 1:n);
Sigma1 = Sigma(1:n, 1:n);
Vcal1 = Vcal(1:s, 1:n);
%Ohat = Ucal1*Sigma1; %alternative
Ohat = Ucal1;

%estimate C and A
Chat = Ohat(1:p, 1:n);
Ahat = pinv(Ohat(1:(r - 1)*p, 1:n))*Ohat((p + 1):r*p, 1:n);


%% Estimate B, D and x0

%Populate Xi matrix
Xi = zeros(Nsignal*p, n);
for i = 1:Nsignal
    if (i == 1)
        Xi((i - 1)*p + (1:p), 1:n) = Chat*Ahat;
    else
        Xi((i - 1)*p + (1:p), 1:n) = Xi((i - 2)*p + (1:p), 1:n)*Ahat;
    end
    
end

%Populate Gamma matrix
Gamma  = zeros(p*Nsignal, n*Nsignal);
for j = 1:Nsignal %iterate cols
    for i = 1:Nsignal %iterate rows
        if (i == 1 && j == 1)
            Gamma((i - 1)*p + (1:p), (j - 1)*n + (1:n)) = Chat;
        elseif (i >= j)
           if (j == 1)
               CA_temp_pow = Xi((i - 2)*p + (1:p), 1:n);
               Gamma((i - 1)*p + (1:p), (j - 1)*n + (1:n)) = CA_temp_pow;
           else
               Gamma((i - 1)*p + (1:p), (j - 1)*n + (1:n)) = Gamma((i - j)*p + (1:p), (1:n));
           end
        end
    end
end
Gamma = Gamma*kron([u0, Usignal(:, 1:(Nsignal - 1))]', eye(n));

Delta = kron(Usignal', eye(p));

Psi = [Xi, Gamma, Delta];

y = reshape(Ysignal, numel(Ysignal), 1);
%solve least squares problem
Theta_hat = (Psi'*Psi)\(Psi'*y);

%extract estimates
x0_hat = Theta_hat(1:n);
Bhat = reshape(Theta_hat(n + (1:n*m)), n, m);
Dhat = reshape(Theta_hat(n + n*m + (1:p*n)), p, n);

%% Validation and comparison

%Estimate system using system id toolbox
data = iddata(Ysignal', Usignal');
[sys_est, x0_est] = n4sid(data, n,'Feedthrough', ones(1, m));

%compare estimates under common coordinate system
C*A*C^-1
Chat*Ahat*Chat^-1
sys_est.C*sys_est.A*sys_est.C^-1

C*B
Chat*Bhat
sys_est.C*sys_est.B

C^-1*x0
sys_est.C^-1*x0_est
Chat^-1*x0_hat

D
Dhat
sys_est.D
