clear all, close all, clc

q = 2;   % number of inputs
p = 2;   % number of outputs
n = 20; % state dimension
r = 5;  % reduced model order

A = [-216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 0.0 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6 108.3 ; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 108.3 -216.6];
B = [108.3 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 0.0 ; 0.0 108.3];
C = [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
%C = eye(20);
D = 0;
sysFullContinuous = ss(A,B,C,D);
%sysFull = ss(A,B,C,D);
sysFull = c2d(sysFullContinuous, 0.01);

%% Plot Hankel singular values
hsvs = hsvd(sysFull); % Hankel singular values
figure
%subplot(1,2,1)
yyaxis right
semilogy(hsvs,'LineWidth',2)
hold on, grid on
%semilogy(r,hsvs(r),'ro','LineWidth',2)
ylim([10^(-15) 1])
ylabel('Hankel singular values')
yyaxis left
barCol = bar(0:length(hsvs),[0; hsvs/sum(hsvs)]);
barCol.FaceColor = 'flat';
barCol.CData(1,:) = [0.9290 0.6940 0.1250];
barCol.CData(2,:) = [0.9290 0.6940 0.1250];
barCol.CData(3,:) = [0.4940 0.1840 0.5560];
barCol.CData(4,:) = [0.4660 0.6740 0.1880];
barCol.CData(5,:) = [0.6350 0.0780 0.1840];
barvalues(barCol, '%0.3f');
ylabel('Fraction of energy captured')

title('Hankel singular values and energy captured by each POD mode');
xlabel('POD modes in descending HSV');
xticks(1:1:20);


%text(1:length(hsvs),hsvs,num2str(hsvs/sum(hsvs)', '%0.3f'),'HorizontalAlignment','center','VerticalAlignment','bottom'); 
%hold on, grid on
%bar(r,sum(hsvs(1:r))/sum(hsvs))
%set(gcf,'Position',[1 1 550 200])
%set(gcf,'PaperPositionMode','auto')

%subplot(1,2,2)
%plot(0:length(hsvs),[0; cumsum(hsvs)/sum(hsvs)],'k','LineWidth',2)
%hold on, grid on
%plot(r,sum(hsvs(1:r))/sum(hsvs),'ro','LineWidth',2)
%set(gcf,'Position',[1 1 550 200])
%set(gcf,'PaperPositionMode','auto')
% print('-depsc2', '-loose', '../figures/FIG_BT_HSVS');

%% Exact balanced truncation
sysBT = balred(sysFull,r);  % balanced truncation

%% Compute BPOD
%[yFull,t,xFull] = impulse(sysFull,0:1:(r*5)+1);
[yFull,t,xFull] = impulse(sysFull,0:0.01:(r*5)+1);
sysAdjoint = ss(sysFull.A',sysFull.C',sysFull.B',sysFull.D',0.01);
%[yAdjoint,t,xAdjoint] = impulse(sysAdjoint,0:1:(r*5)+1);
[yAdjoint,t,xAdjoint] = impulse(sysAdjoint,0:0.01:(r*5)+1);
% not the fastest way to compute, but illustrative
% both xAdjoint and xFull are size m x n x 2
HankelOC = [];  % Compute Hankel matrix H=OC
% we start at 2 to avoid incorporating the D matrix
for i=2:size(xAdjoint,1)
    Hrow = [];
    for j=2:size(xFull,1)
        Ystar = permute(squeeze(xAdjoint(i,:,:)),[2 1]);
        MarkovParameter = Ystar*squeeze(xFull(j,:,:));
        Hrow = [Hrow MarkovParameter];
    end
    HankelOC = [HankelOC; Hrow];
end
[U,Sig,V] = svd(HankelOC);
Xdata = [];
Ydata = [];
for i=2:size(xFull,1)  % we start at 2 to avoid incorporating the D matrix
    Xdata = [Xdata squeeze(xFull(i,:,:))];
    Ydata = [Ydata squeeze(xAdjoint(i,:,:))];
end
Phi = Xdata*V*Sig^(-1/2); % modes
Psi = Ydata*U*Sig^(-1/2);
Ar = Psi(:,1:r)'*sysFull.a*Phi(:,1:r);
Br = Psi(:,1:r)'*sysFull.b;
Cr = sysFull.c*Phi(:,1:r);
Dr = sysFull.d;
sysBPOD = ss(Ar,Br,Cr,Dr,0.01);

sysBPODcontinuous = d2c(sysBPOD, 'tustin');
ArC = sysBPODcontinuous.A
BrC = sysBPODcontinuous.B
CrC = sysBPODcontinuous.C

%% Plot impulse responses for all methods
figure
impulse(sysFull,0:0.01:25), hold on;
impulse(sysBT,0:0.01:25)
impulse(sysBPOD,0:0.01:25)
legend('Full model, n=20','Balanced truncation, r=5','Balanced POD, r=5')

%% Plot impulse responses for all methods
figure
[y1,t1] = impulse(sysFull,0:0.01:200);
[y5,t5] = impulse(sysBPOD,0:0.01:100);
subplot(2,2,1)
stairs(y1(:,1,1),'LineWidth',2);
hold on
stairs(y5(:,1,1),'LineWidth',1.);
ylabel('y_0')
title('u_0')
set(gca,'XLim',[0 60]);
grid on
subplot(2,2,2)
stairs(y1(:,1,2),'LineWidth',2);
hold on
stairs(y5(:,1,2),'LineWidth',1.);
title('u_1')
set(gca,'XLim',[0 60]);
grid on
subplot(2,2,3)
stairs(y1(:,2,1),'LineWidth',2);
hold on
stairs(y5(:,2,1),'LineWidth',1.);
xlabel('t')
ylabel('y_1')
set(gca,'XLim',[0 60]);
grid on
subplot(2,2,4)
stairs(y1(:,2,2),'LineWidth',2);
hold on
stairs(y5(:,2,2),'LineWidth',1.);
xlabel('t')
set(gca,'XLim',[0 60]);
grid on
subplot(2,2,2)
legend('Full model, n=20',['Balanced POD, r=',num2str(r)])
set(gcf,'Position',[100 100 550 350])
set(gcf,'PaperPositionMode','auto')
% % print('-depsc2', '-loose', '../figures/FIG_BT_IMPULSE');