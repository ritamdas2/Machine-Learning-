% EC 414 - HW 4 - Spring 2020
% K-Means starter code

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
figure(1);
hold on;
plot(clust1(:,1),clust1(:,2),'rx')
plot(clust2(:,1),clust2(:,2),'gx')
plot(clust3(:,1),clust3(:,2),'bx')
title('Synthetic Data Set Generation');
xlabel('x1');
ylabel('x2');
%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
nba_data = readmatrix('NBA_stats_2018_2019.xlsx');
figure(4);
hold on;
gscatter(nba_data(:,5), nba_data(:,7));
title('PPG v. MPG NBA 2018-2019');
xlabel('MPG');
ylabel('PPG');

% Problem 4.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 4 folder on Blackboard.

%% K-Means implementation
% Add code below

DATA = [clust1; clust2; clust3];
K = 3;
%MU_init = [3 3; -4 -1; 2 -4]; %part a
%MU_init = [-.14, 2.61; 3.15, -0.84; -3.28, -1.58]; %part b
MU_init = [(rand*2-1)*3, (rand*2-1)*4; (rand*2-1)*3, (rand*2-1)*4;(rand*2-1)*3, (rand*2-1)*4];% (rand*2-1)*3, (rand*2-1)*4; (rand*2-1)*3, (rand*2-1)*4]; %part c
%MU_init = [DATA(5,1), DATA(5,2); DATA(15,1), DATA(15,2);DATA(35,1), DATA(35,2); DATA(55,1), DATA(55,2); DATA(65,1), DATA(65,2)];%DATA(75,1), DATA(75,2); DATA(95,1), DATA(95,2); DATA(115,1), DATA(115,2); DATA(135,1), DATA(135,2); DATA(150,1), DATA(150,2)];
%^^^part c using data points as initial centers to prevent empty clusters

%[DATA, labels] = my_k_means(3, DATA, MU_init);

%function [DATA, labels, WCSS] = my_k_means(K, DATA, MU_init)

% initializations
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
MU_previous = MU_init;
MU_current = MU_init;
labels = ones(length(DATA),1);

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    
   %if isequaln(MU_previous,MU_current) %| (MU_current(1:K,:) - MU_previous(1:K,:)) < convergence_threshold
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        figure(2);
        hold on;
        gscatter(DATA(:,1),DATA(:,2),labels)
        title('Random Center Initialization');
        xlabel('x1');
        ylabel('x2');
        
        %% If converged, get WCSS metric
        % Add code below
        WCSS = zeros(K,1);
        for i = 1:K
            for l = 1:150
                WCSS(i) = sum((DATA(l,:) - MU_current(i,:)).^2);
            end
        end
        WCSS = sum(WCSS);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
KAYS = [2:10];
WCSSVALS = [30.7967520946646, 58.5687955130517, 64.6672572632227, 152.413928894393, 138.330472388503, 134.760727396122, 190.264738560382, 169.541080741177,238.890564127221 ];
figure(3);
hold on;
plot(KAYS, WCSSVALS);
xlabel('K Values');
ylabel('WCSS');
title('WCSS as a function of K');
