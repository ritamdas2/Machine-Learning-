mu = [2,2;-2,2;0,-3.25];
cov1 = 0.02*eye(2);
cov2 = 0.05*eye(2);
cov3 = 0.07*eye(2);
clust1 = mvnrnd(mu(1,:),cov1,50);
clust2 = mvnrnd(mu(2,:),cov2,50);
clust3 = mvnrnd(mu(3,:),cov3,50);
DATA = [clust1; clust2; clust3];


for i = 2:10
    K = i;
for lambda = 15:5:30

ten_rand = zeros(10,K);
for i = 1:10
    ten_rand(i,:) = randperm(size(DATA,1), K);
end
for i = 1:size(ten_rand,1)
    this_mu = zeros(K,2);
    for j = 1:K
        this_mu(j,:) = DATA(ten_rand(i,j),:);
    end
end

MU_init = this_mu;
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

        temp_k = zeros(length(labels), 1);
        temp_d = zeros(size(DATA,1), 1);
        curr_dist = zeros(size(DATA,1), 1);
        WCSS = 0;

        for j=1:K 
            temp_k = labels == j;
            temp_d = DATA(temp_k,:);
            curr_dist = pdist2(temp_d, MU_current(j,:));
            curr_dist = curr_dist.^2;
            WCSS = WCSS + sum(curr_dist) + (lambda*K);
        end
        
        fprintf("LAMBDA: %d\n", lambda);
        fprintf("K: %d\n", K);
        fprintf("WCSS: %.3f\n" , WCSS);
        
        figure(1);
        hold on;
        plot(K-1, WCSS, 'd');
        txt = {'Steepest Slope is lambda = 30'};
        text(3,3000,txt)
        txt2 = {'Next Steepest Slope is lambda = 25','and so on until lambda = 15'};
        text(3,2500,txt2)
        
        xlabel('K Values');
        ylabel('WCSS');
        xlim([1 10]);
        title('WCSS = f(k,lambda) as a function of K');
        hold off;
    end
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%END 4.3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%