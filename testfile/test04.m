X=A(B>2);
X1=A1(B1>2);
plot(X);
hold on;
t=500:1501;
disp(sum((X(t)-X1).^2))
plot(t,X1);