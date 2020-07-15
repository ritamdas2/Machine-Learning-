% EC 414 Introduction to Machine Learning
% Spring semester, 2020
% Homework 3
% by Ritam Das
%
% Problem 4.3 Nearest Neighbor Classifier
% a), b), c), and d)

clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()
col1 = Xtrain(:,1);
col2 = Xtrain(:,2);
gscatter(col1, col2, ytrain,'rgb')

% label axis and include title
xlabel('Column 1')
ylabel('Column 2')
title('Scatter Plot of All Training Data')


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute probabilities of being in class 2 for each point on grid
%p10kNN(2|x)
distances = zeros(200,2);
probabilities = zeros(Ntest,1);
for i = 1:Ntest
    count = 0;
    for j = 1:200
        distances(j,1) = sqrt((Xgrid(i)-Xtrain(j,1))^2 + (Ygrid(i)-Xtrain(j,2))^2);
        distances(j,2) = ytrain(j);
    end
    distances = sortrows(distances,1);
    for l = 1:10
        if distances(l,2) == 2
            count = count + 1;
        end
    end
   probabilities(i) = count/10;
end


% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Column 1')
ylabel('Column 2')
title('Colormap of Prob. of Class 2')


% repeat steps above for class 3 below
for i = 1:Ntest
    count = 0;
    for j = 1:200
        distances(j,1) = sqrt((Xgrid(i)-Xtrain(j,1))^2 + (Ygrid(i)-Xtrain(j,2))^2);
        distances(j,2) = ytrain(j);
    end
    distances = sortrows(distances,1);
    for l = 1:10
        if distances(l,2) == 3
            count = count + 1;
        end
    end
   probabilities(i) = count/10;
end

% Figure for class 3
figure
class3ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Column 1')
ylabel('Column 2')
title('Colormap of Prob. of Class 3')

%% c) Class label predictions
% K = 1 case

% compute predictions 
for i = 1:Ntest
    count = 0;
    for j = 1:200
        distances(j,1) = sqrt((Xgrid(i)-Xtrain(j,1))^2 + (Ygrid(i)-Xtrain(j,2))^2);
        distances(j,2) = ytrain(j);
    end
    distances = sortrows(distances,1);
    ypred(i) = distances(1,2);
end
    
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('Column 1')
ylabel('Column 2')
title('Class Prediction k = 1')

% repeat steps above for the K=5 case. Include code for this below.

for i = 1:Ntest
    class1Count = 0;
    class2Count = 0;
    class3Count = 0;
    for j = 1:200
        distances(j,1) = sqrt((Xgrid(i)-Xtrain(j,1))^2 + (Ygrid(i)-Xtrain(j,2))^2);
        distances(j,2) = ytrain(j);
    end
    distances = sortrows(distances,1);
    for l = 1:5
     if distances(l,2) == 1
        class1Count = class1Count + 1;
     elseif distances(l,2) == 2
        class2Count = class2Count + 1;
     elseif distances(l,2) == 3
        class3Count = class3Count + 1;
     end
    end
    countsArr = [class1Count 1; class2Count 2; class3Count 3];
    countsArr = sortrows(countsArr,1);
    ypred(i) = countsArr(3,2);
end
    
figure(5);
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('Column 1')
ylabel('Column 2')
title('Class Prediction k = 5')


%% d) LOOCV CCR computations
%determine leave-one-out predictions for k
ypred = zeros(200,1);
distances = zeros(Ntrain,2);
for k = 1:2:11
    for i= 1:Ntrain
        for j = 1:Ntrain
            distances(j,1) = sqrt((Xtrain(j,1)-Xtrain(i,1))^2+ (Xtrain(j,2)-Xtrain(i,2))^2);
            distances(j,2) = ytrain(j);
        end
        distances = sortrows(distances);
        ypred(i) = mode(distances(2:k+1,2));
    end

    % compute confusion matrix
    conf_mat = confusionmat(ytrain(:), ypred(:));
    % from confusion matrix, compute CCR
    CCR = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3))/200;
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end


% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
figure(6);
kays = [1:2:11];
plot(kays,CCR_values,'rx');
ylim([.65, .9]);
xlabel('k-value');
ylabel('LOOCV CCR');
title('LOOCV CCR for each k value')

