% This script demos a particle filter and is split into four sections with
% slightly different function

%% Section 1: 
% Particle filter assumes a linear transition model (so a Kalamn filter
% could be used instead)

clc
clear
close all

dt = 6; %timestep
A = [1, 0, 0, dt, 0, 0;
    0, 1, 0, 0, dt, 0;
    0, 0, 1, 0, 0, dt;
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1]; %state update matrix
    
y0 = [0; 0; 9000; -100; 115; 0]; %initial state
y = y0;

kmax = 120; %number of timesteps to estimate

%generate the true trajectory
%the aircraft will descend at a constant rate (with some noise) until it
%reasches 2000m 
for k = 1:kmax
    noise = mvnrnd([2; 2; 0; -0.1; 0.1; 0.01], diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 0.01^2]))';
    if (y(3, k) > 2000)
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + [0; 0; 0; 0; 0; -12] + noise];
    else
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + noise];
    end
    
end

scatter3(y(1, :), y(2, :), y(3, :))
hold on

N = 1000; %number of particles
%generate inital particles
x = repmat(y0, 1, N) + [250*randn(1, N); 250*randn(1, N); 250*randn(1, N); 
    10*randn(1, N); 10*randn(1, N); 1*randn(1, N)];

x_est = []; %store state point estimates

%perform estimates
for k = 1:(kmax - 1)
    %propagate particles
    x = A*x + mvnrnd(zeros(N, 6), diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 1^2]))';
    
    %generate noisy measurement
    z = y(:, k + 1) + mvnrnd([7; 7; 3; -0.1; 0.1; 0.01], diag([10^2, 10^2, 2^2, 0.1^2, 0.1^2, 0.01^2]))';
    
    %Gaussian likelihood not used because of numerical issues
    %w = mvnpdf(x', repmat(z', N, 1), diag([50^2, 50^2, 50^2, 20^2, 20^2, 10^2]));
    %use a reciprocal of squared weighted norm likelihood
    w = 1./sum((diag([0.3, 0.3, 1, 0.2, 0.2, 1])*(x - repmat(z, 1, N))).^2, 1);
    %normalise weights
    w = w/sum(w);
    %bootstrap resampling
    [~, I] = histc(rand(1, N), cumsum(w));
    x = x(:, I + 1);

    scatter3(x(1, :), x(2, :), x(3, :), 'g.', 'MarkerEdgeAlpha',.7)
    %point estimate using mean of particles
    x_est = [x_est, mean(x, 2)];
end

hold on
scatter3(x_est(1, :), x_est(2, :), x_est(3, :), '.')

%% Section 2: 
% Particle filter assumes a nonlinear transition model where aircraft
% descends at a constant velocity until it reaches a deadband around its
% desired altitude. Additionally, the particle is augmented with another
% state which adaptively estimates the descent rate

clc
clear
close all


dt = 6; %timestep
A = [1, 0, 0, dt, 0, 0;
    0, 1, 0, 0, dt, 0;
    0, 0, 1, 0, 0, dt;
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1]; %state update matrix
    
y0 = [0; 0; 9000; -100; 115; 0]; %initial state
y = y0;

kmax = 120; %number of timesteps to estimate

%generate the true trajectory
%the aircraft will descend at a constant rate (with some noise) until it
%reasches 2000m 
for k = 1:kmax
    noise = mvnrnd([2; 2; 0; -0.1; 0.1; 0.01], diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 0.01^2]))';
    if (y(3, k) > 2000)
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + [0; 0; 0; 0; 0; -12] + noise];
    else
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + noise];
    end
    
end

scatter3(y(1, :), y(2, :), y(3, :))
hold on

N = 1000; %number of particles
%generate initial particles
x = repmat(y0, 1, N) + [250*randn(1, N); 250*randn(1, N); 250*randn(1, N); 
    10*randn(1, N); 10*randn(1, N); 1*randn(1, N)];
x = [x; 10.5 + 1.2*randn(1, N)]; %inital guesses of descent rate

x_est = []; %store state point estimates

for k = 1:(kmax - 1)
    %propagate particles
    for i = 1:N
        if (x(3, i) > 2050)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; -x(7, i)];
        elseif (x(3, i) < 1950)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; x(7, i)];
        else
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i);
        end
    end
    x = x + mvnrnd(zeros(N, 7), diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 1^2, 0.02^2]))';
    
    %generate noisy measurement
    z = y(:, k + 1) + mvnrnd([7; 7; 3; -0.1; 0.1; 0.01], diag([10^2, 10^2, 2^2, 0.1^2, 0.1^2, 0.01^2]))';
    
    %likelihood function is reciprocal of squared weigthed norm
    w = 1./sum((diag([0.5, 0.5, 1, 0.2, 0.2, 1])*(x(1:6, :) - repmat(z, 1, N))).^2, 1);
    %normalise weights
    w = w/sum(w);
    %bootstrap resampling
    [~, I] = histc(rand(1, N), cumsum(w));
    x = x(:, I + 1);

    scatter3(x(1, :), x(2, :), x(3, :), 'g.')
    %point estimate using mean of particles
    x_est = [x_est, mean(x, 2)];
end

hold on
scatter3(x_est(1, :), x_est(2, :), x_est(3, :), '.')

%% Section 3:
% Transition distribution is used to predict ahead the future trajectory
% from only the first timestep

clc
clear
close all


dt = 6; %timestep
A = [1, 0, 0, dt, 0, 0;
    0, 1, 0, 0, dt, 0;
    0, 0, 1, 0, 0, dt;
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1]; %state update matrix
    
y0 = [0; 0; 9000; -100; 115; 0]; %initial state
y = y0;

kmax = 120; %number of timesteps to estimate

%generate the true trajectory
%the aircraft will descend at a constant rate (with some noise) until it
%reasches 2000m 
for k = 1:kmax
    noise = mvnrnd([2; 2; 0; -0.1; 0.1; 0.01], diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 0.01^2]))';
    if (y(3, k) > 2000)
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + [0; 0; 0; 0; 0; -12] + noise];
    else
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + noise];
    end
    
end

scatter3(y(1, :), y(2, :), y(3, :))
hold on

N = 1000; %number of particles
%generate initial particles
x = repmat(y0, 1, N) + [250*randn(1, N); 250*randn(1, N); 250*randn(1, N); 
    10*randn(1, N); 10*randn(1, N); 1*randn(1, N)];
x = [x; 10.5 + 1.2*randn(1, N)]; %inital guesses of descent rate

x_est = []; %store state point estimates

%perform predictions
for k = 1:(kmax - 1)
    %propagate particles
    for i = 1:N
        if (x(3, i) > 2050)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; -x(7, i)];
        elseif (x(3, i) < 1950)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; x(7, i)];
        else
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i);
        end
    end
    x = x + mvnrnd(zeros(N, 7), diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 1^2, 0.02^2]))';

    scatter3(x(1, :), x(2, :), x(3, :), 'g.', 'MarkerEdgeAlpha',.2)
    x_est = [x_est, mean(x, 2)];
end

hold on
scatter3(x_est(1, :), x_est(2, :), x_est(3, :), 'filled')

%% Section 4:
% The transition distribution is used to predict the future trajectory
% while simultaneous filtering; this is played as an animation

clc
clear
close all


dt = 6; %timestep
A = [1, 0, 0, dt, 0, 0;
    0, 1, 0, 0, dt, 0;
    0, 0, 1, 0, 0, dt;
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1]; %state update matrix
    
y0 = [0; 0; 9000; -100; 115; 0]; %initial state
y = y0;

kmax = 120; %number of timesteps to estimate

%generate the true trajectory
%the aircraft will descend at a constant rate (with some noise) until it
%reasches 2000m 
for k = 1:kmax
    noise = mvnrnd([2; 2; 0; -0.1; 0.1; 0.01], diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 0.01^2]))';
    if (y(3, k) > 2000)
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + [0; 0; 0; 0; 0; -12] + noise];
    else
        y = [y, [A(1:5, :); zeros(1, 6)]*y(:, k) + noise];
    end
    
end

scatter3(y(1, :), y(2, :), y(3, :))
hold on

N = 1000; %number of particles
%generate initial particles
x = repmat(y0, 1, N) + [250*randn(1, N); 250*randn(1, N); 250*randn(1, N); 
    10*randn(1, N); 10*randn(1, N); 1*randn(1, N)];
x = [x; 10.5 + 1.2*randn(1, N)]; %inital guesses of descent rate

x_est = []; %store state point estimates

%perform filtering
for kk = 1:(kmax - 1)
    clf
    scatter3(y(1, :), y(2, :), y(3, :))
    hold on
    
    %propagate particles
    for i = 1:N
        if (x(3, i) > 2050)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; -x(7, i)];
        elseif (x(3, i) < 1950)
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i) + [0; 0; 0; 0; 0; x(7, i)];
        else
            x(1:6, i) = [A(1:5, :); zeros(1, 6)]*x(1:6, i);
        end
    end
    x = x + mvnrnd(zeros(N, 7), diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 1^2, 0.02^2]))';
    
    %generate noisy measurement
    z = y(:, kk + 1) + mvnrnd([0; 0; 0; 0; 0; 0], diag([10^2, 10^2, 2^2, 0.1^2, 0.1^2, 0.01^2]))';
    
    %likelihood function is reciprocal of squared weighted norm
    w = 1./sum((diag([0.5, 0.5, 1, 0.1, 0.1, 1])*(x(1:6, :) - repmat(z, 1, N))).^2, 1);
    %normalise weights
    w = w/sum(w);
    %bootstrap resampling
    [~, I] = histc(rand(1, N), cumsum(w));
    x = x(:, I + 1);
    %point estimate using mean of particles
    x_est = [x_est, mean(x, 2)];
    
    scatter3(x(1, :), x(2, :), x(3, :), 'g.', 'MarkerEdgeAlpha',.3)
    scatter3(x_est(1, 1:kk), x_est(2, 1:kk), x_est(3, 1:kk), 'filled')
    
    
    xp = x; %make a copy of particles
    %perform prediction
    for k = (kk + 1):(kmax - 1)
        %propagate particles
        for i = 1:N
            if (xp(3, i) > 2050)
                xp(1:6, i) = [A(1:5, :); zeros(1, 6)]*xp(1:6, i) + [0; 0; 0; 0; 0; -xp(7, i)];
            elseif (xp(3, i) < 1950)
                xp(1:6, i) = [A(1:5, :); zeros(1, 6)]*xp(1:6, i) + [0; 0; 0; 0; 0; xp(7, i)];
            else
                xp(1:6, i) = [A(1:5, :); zeros(1, 6)]*xp(1:6, i);
            end
        end
        xp = xp + mvnrnd(zeros(N, 7), diag([2^2, 2^2, 2^2, 0.01^2, 0.01^2, 1^2, 0.02^2]))';
        
        scatter3(xp(1, :), xp(2, :), xp(3, :), 'g.', 'MarkerEdgeAlpha',.3)
        
    end
   
    drawnow
    axis([0 100000 -80000 0 0 10000])
    
end

