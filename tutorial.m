a = ones(5,3);
M = magic(3); % magic square
I = eye(b); %identity matrix
y = linspace(a,n,b);
%Create a vector of 7 evenly spaced points in the interval [-5,5].
y1 = linspace(-5,5,7);
y=[3:2:9];
[m,n] = size(B);
lambdas = eig(A); %eigenvalues
%A*x = b;
%x = inv(A)*b;
%use pseudo inverse -- pinv(A)
x = pinv(A)*b;