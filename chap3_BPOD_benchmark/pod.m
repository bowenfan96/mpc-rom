A = [-.75 1; -.3 -.75];
B = [2; 1];
C = [1 2];
D = 0;
sysFull = ss(A,B,C,D);
r = 10;

Wc = gram(sysFull,'c'); % Controllability Gramian
Wo = gram(sysFull,'o'); % Observability Gramian
[sysb,g,Ti,T] = balreal(sysFull); % Balance the system
BWc = gram(sysb,'c') % Balanced Gramians
BWo = gram(sysb,'o')

hsvs = hsvd(sysFull); % Hankel singular values

[yFull,t,xFull] = impulse(sysFull,0:1:(r*5)+1);
sysAdj = ss(sysFull.A',sysFull.C',sysFull.B',sysFull.D',-1);
[yAdj,t,xAdj] = impulse(sysAdj,0:1:(r*5)+1);
sysBPOD = BPOD(sysFull,sysAdj,r)

% Not the fastest way to compute, but illustrative
% Both xAdj and xFull are size m x n x 2

HankelOC = []; % Compute Hankel matrix H=OC

for i=2:size(xAdj,1) % Start at 2 to avoid the D matrix
Hrow = [];
for j=2:size(xFull,1)
Ystar = permute(squeeze(xAdj(i,:,:)),[2 1]);
MarkovParameter = Ystar*squeeze(xFull(j,:,:));
Hrow = [Hrow MarkovParameter];
end
HankelOC = [HankelOC; Hrow];
end
[U,Sig,V] = svd(HankelOC);
Xdata = [];
Ydata = [];
for i=2:size(xFull,1) % Start at 2 to avoid the D matrix
Xdata = [Xdata squeeze(xFull(i,:,:))];
Ydata = [Ydata squeeze(xAdj(i,:,:))];
end
Phi = Xdata*V*Sig^(-1/2);
Psi = Ydata*U*Sig^(-1/2);
Ar = Psi(:,1:r)'*sysFull.a*Phi(:,1:r);
Br = Psi(:,1:r)'*sysFull.b;
Cr = sysFull.c*Phi(:,1:r);
Dr = sysFull.d;
sysBPOD = ss(Ar,Br,Cr,Dr,-1);
