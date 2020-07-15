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

        temp_k = zeros(length(labels), 1);
        temp_d = zeros(size(DATA,1), 1);
        new_dist = zeros(size(DATA,1), 1);
        WCSS = 0;
        for j=1:K
            temp_k = labels == j;
            temp_d = DATA(temp_k,:);
            new_dist = pdist2(temp_d, MU_current(j,:));
            new_dist = new_dist.^2;
            WCSS = WCSS + sum(new_dist);
        end
        fprintf("WCSS: %.3f\n", WCSS);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%END C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%