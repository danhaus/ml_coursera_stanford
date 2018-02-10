load('ex3data1.mat');
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);