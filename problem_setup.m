clc;
clear all;
close all;

%% ================= STEP 2: Initialize Parameters =================
[n, l, k_max, tol] = init_parameters();

%% ================= STEP 3: Generate System =================
[A, b, x] = generate_system(n);

%% ================= STEP 4: Initial Residual =================
r = b - A*x;          % Initial residual vector
beta = norm(r);       % Residual norm

fprintf('STEP 4: Initial Residual Norm (beta): %e\n', beta);

% Plot initial residual vector
figure;
plot(r, 'b');
xlabel('Index');
ylabel('Residual Value');
title('STEP 4: Initial Residual Vector r = b - Ax_0');
grid on;

%% ================= STEP 5: Krylov Subspace (Arnoldi Process) =================
V = zeros(n, k_max);
H = zeros(k_max, k_max);
V(:,1) = r / beta;

for j = 1:k_max
    w = A * V(:,j);
    for i = 1:j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);
    if j < k_max && H(j+1,j) ~= 0
        V(:,j+1) = w / H(j+1,j);
    end
end

% Plot first Krylov basis vector
figure;
plot(V(:,1));
xlabel('Index');
ylabel('Value');
title('STEP 5: First Krylov Basis Vector v_1');
grid on;

%% ================= STEP 6: Apply Randomized Sketching =================
S = randn(l, n);  % Gaussian sketch matrix
SA = S * A;
SV = S * V;
Sb = S * b;

% Plot histogram of sketch matrix values
figure;
histogram(S(:), 30);
xlabel('Value');
ylabel('Frequency');
title('STEP 6: Sketch Matrix Distribution (S)');
grid on;

%% ================= STEP 7: Solve Reduced System =================
y = (SV' * SA * V) \ (SV' * Sb);

% Plot y vector
figure;
plot(y, '-o');
xlabel('Index');
ylabel('Value');
title('STEP 7: Solution of Reduced System y');
grid on;

%% ================= STEP 8: Update Solution =================
x = x + V * y;

% Plot updated solution vector
figure;
plot(x);
xlabel('Index');
ylabel('Value');
title('STEP 8: Updated Solution Vector x');
grid on;

%% ================= STEP 9-10: Iterative Loop =================
residuals = [];
tic;  % start execution time

for iter = 1:k_max
    
    % Arnoldi Process
    r = b - A*x;
    beta = norm(r);
    
    V = zeros(n, k_max);
    H = zeros(k_max, k_max);
    V(:,1) = r / beta;
    
    for j = 1:k_max
        w = A * V(:,j);
        for i = 1:j
            H(i,j) = V(:,i)' * w;
            w = w - H(i,j) * V(:,i);
        end
        H(j+1,j) = norm(w);
        if j < k_max && H(j+1,j) ~= 0
            V(:,j+1) = w / H(j+1,j);
        end
    end
    
    % Randomized Sketching
    S = randn(l, n);
    SV = S * V;
    SA = S * A;
    Sb = S * b;
    
    % Solve reduced system
    y = (SV' * SA * V) \ (SV' * Sb);
    
    % Update solution
    x = x + V * y;
    
    % Compute residual
    res = norm(b - A*x);
    residuals = [residuals res];
    
    fprintf('Iteration %d, Residual = %e\n', iter, res);
    
    % Check convergence after update
    if res < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

exec_time = toc;  % execution time
fprintf('Total Execution Time: %.2f seconds\n', exec_time);

%% ================= STEP 11: Plot Convergence =================
figure;
plot(1:length(residuals), residuals, '-o', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Residual Norm ||r_k||');
title('STEP 11: Residual Convergence Plot');
grid on;

%% ==================== FUNCTIONS ====================
function [n, l, k_max, tol] = init_parameters()
    n = 2500;
    l = round(n/4);
    k_max = 50;
    tol = 1e-6;
end

function [A, b, x] = generate_system(n)
    A = sprand(n, n, 0.01);      % Sparse random matrix
    A = A + 10*speye(n);         % Ensure stability and better conditioning
    b = randn(n,1);              % Random output vector
    x = zeros(n,1);              % Initial guess
end