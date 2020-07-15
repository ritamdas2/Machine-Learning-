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
figure(5);
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
MU_init = [3 3; -4 -1; 2 -4]; %part a
%MU_init = [-.14, 2.61; 3.15, -0.84; -3.28, -1.58]; %part b

%^^^part c using data points as initial centers to prevent empty clusters


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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure(2);
        hold on;
        gscatter(DATA(:,1),DATA(:,2),labels)
        title('Given Center Initialization');
        xlabel('x1');
        ylabel('x2');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%END A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


DATA = [clust1; clust2; clust3];
K = 3;
%MU_init = [3 3; -4 -1; 2 -4]; %part a
MU_init = [-.14, 2.61; 3.15, -0.84; -3.28, -1.58]; %part b



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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure(2);
        hold on;
        gscatter(DATA(:,1),DATA(:,2),labels)
        title('Given Center Initialization');
        xlabel('x1');
        ylabel('x2');
    end
    
    %%%%%%%%%%%%%%%%%%%%%END B%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATA = [clust1; clust2; clust3];
K = 3;

ten_rand_mu = zeros(10,K);
for i = 1:10
    ten_rand_mu(i,:) = randperm(size(DATA,1), K);
end
for i = 1:size(ten_rand_mu,1)
    curr_mu = zeros(K,2);
    for j = 1:K
        curr_mu(j,:) = DATA(ten_rand_mu(i,j),:);
    end
end

MU_init = curr_mu;
%^^^part c using data points as initial centers to prevent empty clusters


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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   %if isequaln(MU_previous,MU_current) %| (MU_current(1:K,:) - MU_previous(1:K,:)) < convergence_threshold
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        figure(3);
        hold on;
        gscatter(DATA(:,1),DATA(:,2),labels)
        title('Given Center Initialization');
        xlabel('x1');
        ylabel('x2');

        k_t = zeros(length(labels), 1);
        data_t = zeros(size(DATA,1), 1);
        new_dist = zeros(size(DATA,1), 1);
        WCSS = 0;
        for j=1:K
            k_t = labels == j;
            data_t = DATA(k_t,:);
            new_dist = pdist2(data_t, MU_current(j,:));
            new_dist = new_dist.^2;
            WCSS = WCSS + sum(new_dist);
        end
        fprintf("WCSS: %.3f\n", WCSS);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%END C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


DATA = [clust1; clust2; clust3];

for i = 1:10
    K = i;


ten_rand_mu = zeros(10,K);
for i = 1:10
    ten_rand_mu(i,:) = randperm(size(DATA,1), K);
end
for i = 1:size(ten_rand_mu,1)
    curr_mu = zeros(K,2);
    for j = 1:K
        curr_mu(j,:) = DATA(ten_rand_mu(i,j),:);
    end
end

MU_init = curr_mu;
%^^^part c using data points as initial centers to prevent empty clusters


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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   %if isequaln(MU_previous,MU_current) %| (MU_current(1:K,:) - MU_previous(1:K,:)) < convergence_threshold
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')

        k_t = zeros(length(labels), 1);
        data_t = zeros(size(DATA,1), 1);
        new_dist = zeros(size(DATA,1), 1);
        WCSS = 0;
        for j=1:K
            k_t = labels == j;
            data_t = DATA(k_t,:);
            
            new_dist = pdist2(data_t, MU_current(j,:));
            new_dist = new_dist.^2;
            WCSS = WCSS + sum(new_dist);
        end
        fprintf("WCSS: %.3f\n", WCSS);
        fprintf("K: %d\n", K);
        figure(4);
        hold on;
        plot(K-1, WCSS, 'd');
        xlabel('K Values');
        ylabel('WCSS');
        xlim([1 10]);
        
        title('WCSS as a function of K');
        hold off;
    end
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%END D%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATA = [nba_data(:,5), nba_data(:,7)];

K = 10;

ten_rand_mu = zeros(10,K);
for i = 1:10
    ten_rand_mu(i,:) = randperm(size(DATA,1), K);
end
for i = 1:size(ten_rand_mu,1)
    curr_mu = zeros(K,2);
    for j = 1:K
        curr_mu(j,:) = DATA(ten_rand_mu(i,j),:);
    end
end

MU_init = curr_mu;
%^^^part c using data points as initial centers to prevent empty clusters


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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')

        k_t = zeros(length(labels), 1);
        data_t = zeros(size(DATA,1), 1);
        new_dist = zeros(size(DATA,1), 1);
        WCSS = 0;
        for j=1:K
            k_t = labels == j;
            data_t = DATA(k_t,:);
            new_dist = pdist2(data_t, MU_current(j,:));
            new_dist = new_dist.^2;
            WCSS = WCSS + sum(new_dist);
        end
        figure(6);
        hold on;
       gscatter(DATA(:,1), DATA(:,2), labels)
       title('NBA DATA Clustering');
       xlabel('x1');
        ylabel('x2');
    end
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%END E%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rings = sample_circle(3);
DATA = rings;

K = 3;


ten_rand_mu = zeros(10,K);
for i = 1:10
    ten_rand_mu(i,:) = randperm(size(DATA,1), K);
end
for i = 1:size(ten_rand_mu,1)
    curr_mu = zeros(K,2);
    for j = 1:K
        curr_mu(j,:) = DATA(ten_rand_mu(i,j),:);
    end
end

MU_init = curr_mu;
%^^^part c using data points as initial centers to prevent empty clusters


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

    dist = pdist2(DATA, MU_current); % distances between 2 sets --> data set and current mean set
    [~, labels] = min(dist, [], 2); % store output of labels
    
    MU_previous = MU_current;
    index = zeros(length(labels), 1);
    for j = 1:K
        index = labels == j;
        MU_current(j,:) = mean(DATA(index,:));
    end
    
 
   if (all(diag(pdist2(MU_current, MU_previous)))) < convergence_threshold
        converged=1;
   end
    if (converged == 1)
        fprintf('\nConverged.\n')

        k_t = zeros(length(labels), 1);
        data_t = zeros(size(DATA,1), 1);
        new_dist = zeros(size(DATA,1), 1);
        WCSS = 0;
        for j=1:K
            k_t = labels == j;
            data_t = DATA(k_t,:);
            new_dist = pdist2(data_t, MU_current(j,:));
            new_dist = new_dist.^2;
            WCSS = WCSS + sum(new_dist);
        end
        figure(7);
        hold on;
       gscatter(DATA(:,1), DATA(:,2), labels)
       title('Using Sample Circle for Data Set Generation');
       xlabel('x1');
       ylabel('x2');
       figure(8);
       gscatter(DATA(:,1), DATA(:,2))
       title('Using Sample Circle for Data Set Generation');
       xlabel('x1');
       ylabel('x2');
    end
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FINALLY
%%%%%%%%%%%%%%%%%%%%%%%%%%END F%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%