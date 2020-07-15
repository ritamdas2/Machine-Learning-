
clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);
ypred = zeros(200,1);
dist = zeros(Ntrain,2);
dist(:,1) = Xtrain(:,1);
dist(:,2) = Xtrain(:,2);
dist(:,3) = ytrain(:,1);
figure(6);
hold on;
 
for k = 1:2:11
    for j= 1:Ntrain
        for i = 1:Ntrain

dist(i,4) = sqrt((dist(j,1) - Xtrain(i,1))^2 + (dist(j,2) - Xtrain(i,2)).^2); %calculating all the distances
if (i==200) %after we calculate all the distances to the point               
ten_closest_dist = mink(dist(:,4),k+1); %find the smallest K distances in the array
ten_closest_dist(1,:)=[ ]; %empty out the 0 distance
for m = 1:k
index = find(dist(:,4) == ten_closest_dist(m,1));% leave out point on itself
ten_closest_dist(m,2) = dist(index,3);
ypred(j,1) = mode(ten_closest_dist(:,2));

end
end
end
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
plot(x,CCR_values,'*','MarkerSize',15);
xlabel('k-value');
ylabel('LOOCV CCR');
title('LOOCV CCR for each k value')
hold off;