
clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);
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

% determine leave-one-out predictions for k
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
x=1:2:11;
plot(x,CCR_values,'x');
ylim([.65, .9]);
xlabel('k-value');
ylabel('LOOCV CCR');
title('LOOCV CCR for each k value')
hold off;