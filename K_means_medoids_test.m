close all;
% set algorithm parameters
TOL = 0.0004;
ITER = 30;
kappa = 4;
method='k_meds';

% generate random data
X = [1000*randn(1000,2) + 1000; 2000*randn(1000,2) + 5000];

% run k-Means on random data
tic;
[C, I, iter] = K_means_medoids(X, kappa, ITER, TOL, method);
toc

% show number of iteration taken by k-means
disp([method ' instance took ' int2str(iter) ' iterations to complete']);

% available colos for the points in the resulting clustering plot
colors = {'red', 'green', 'blue', 'black'};

% show plot of clustering
figure(2);
for i=1:kappa
   %find(I == i)
   hold on, plot(X(find(I == i), 1), X(find(I == i), 2), '.', 'color', colors{i});
end
title(method);
hold on;