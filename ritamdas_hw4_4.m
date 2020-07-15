% EC 414 - HW 4 - Spring 2020
% DP-Means starter code

clear, clc, close all,

%% Generate Gaussian data:
% Add code below:
mu = [2,2;-2,2;0,-3.25];
cov1 = 0.02*eye(2);
cov2 = 0.05*eye(2);
cov3 = 0.07*eye(2);
clust1 = mvnrnd(mu(1,:),cov1,50);
clust2 = mvnrnd(mu(2,:),cov2,50);
clust3 = mvnrnd(mu(3,:),cov3,50);
DATA = [clust1; clust2; clust3];

%% Generate NBA data:
% Add code below:
nba_data = readmatrix('NBA_stats_2018_2019.xlsx');
% HINT: readmatrix might be useful here
%% DP Means method:

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
num_points = length(DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
L = {};
L = [L [1:num_points]];

% Class indicators/labels
Z = ones(1,num_points);

% means
MU = [];
MU = [MU; mean(DATA,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations for algorithm:
converged = 0;
t = 0;
while (converged == 0)
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %% Per Data Point:
    for i = 1:num_points
        
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
        % Write code below here:
        dist = zeros(K,1);
        for j = 1:K
            dist = pdist2(DATA, MU(j,:));
        end
        
        %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
        % Write code below here:
        if (min(dist)^2) > LAMBDA
            K = K + 1;
            Z(i) = K;
            MU(K,1) = DATA(i,1);
            MU(K,2) = DATA(i,2);
        end

    end
    
    %% CODE 3 - Form new sets of points (clusters)
    % Write code below here:
    for i = 1:num_points
        dist = zeros(K,1);
        for j = 1:K
            dist = pdist2(DATA, MU(j,:));
        end
        index = find(dist == min(dist));
        Z(i) = index;
    end
    %% CODE 4 - Recompute means per cluster
    % Write code below here:
    
    for j = 1:K
        MU(j,:) = mean(DATA(j,:));
    end
    
    %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
    % Write code below here:
    if (t > 1 & (MU(K,1) - MU(K,2))) < convergence_threshold
        converged=1;
    end
    
    %% CODE 6 - Plot final clusters after convergence 
    % Write code below here:
    
    if (converged)
        %%%%
            gscatter(DATA(:,1), DATA(:,2), Z)
            title('4.4 PART B')
            xlabel('x1');
            ylabel('x2');
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATA = [nba_data(:,5), nba_data(:,7)];

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
num_points = length(DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
L = {};
L = [L [1:num_points]];

% Class indicators/labels
Z = ones(1,num_points);

% means
MU = [];
MU = [MU; mean(DATA,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations for algorithm:
converged = 0;
t = 0;
while (converged == 0)
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %% Per Data Point:
    for i = 1:num_points
        
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
        % Write code below here:
        dist = zeros(K,1);
        for j = 1:K
            dist = pdist2(DATA, MU(j,:));
        end
        
        %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
        % Write code below here:
        if (min(dist)^2) > LAMBDA
            K = K + 1;
            Z(i) = K;
            MU(K,1) = DATA(i,1);
            MU(K,2) = DATA(i,2);
        end

    end
    
    %% CODE 3 - Form new sets of points (clusters)
    % Write code below here:
    for i = 1:num_points
        dist = zeros(K,1);
        for j = 1:K
            dist = pdist2(DATA, MU(j,:));
        end
        index = find(dist == min(dist));
        Z(i) = index;
    end
    %% CODE 4 - Recompute means per cluster
    % Write code below here:
    
    for j = 1:K
        MU(j,:) = mean(DATA(j,:));
    end
    
    %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
    % Write code below here:
    if (t > 1 & (MU(K,1) - MU(K,2))) < convergence_threshold
        converged=1;
    end
    
    %% CODE 6 - Plot final clusters after convergence 
    % Write code below here:
    
    if (converged)
        %%%%
            gscatter(DATA(:,1), DATA(:,2), Z)
            title('4.4 PART B using NBA Data')
            xlabel('x1');
            ylabel('x2');
    end    
end
