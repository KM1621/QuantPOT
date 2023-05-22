%% Plotting uniform
% clear all
clc
%close all

x = -1:0.001:1;
y = sign(x).*2.^log2(abs(round(x*2)/2));
y_appr = -1+0.5*(logsig(100*(x+1*0.25)) + logsig(100*(x+0.75))) + 0.5*(logsig(100*(x-1*0.25)) + logsig(100*(x-0.75)));
%y_appr = 0.1*sin(10*(x-0.1)) + x;
figure
plot(x, y,'LineWidth',5)
%legend('$Q_{uniform}$','Interpreter','latex') 
hold on 
plot(x, y_appr,'LineWidth',5)
Df = diff(y_appr);
%legend('Projection','Projection logsig approximation')

%legend(['$2^{\psi}$','Interpreter','latex', '$Q_{uniform}$','Interpreter','latex']) 
legend(['$Q_{uniform}$'],  ['$Q_{appr} = -1+0.5*\sum_{i=0}^{N-1} logsig(a_i \times (x+b_i \times 0.25 ))$'],'Interpreter','latex') 
set(gca,'FontSize',18, 'fontWeight','bold')

%% Plotting APOT
% clear all
clc
%close all

x = -1:0.001:1;
v=x;
y = sign(x).*2.^log2(abs(round(x*2)/2));
y_POT_targets = [-2^0 -2^-1 -2^-2 0 2^-2 2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_POT_targets),y_POT_targets,0.5 + (1:numel(y_POT_targets)-1)) Inf]);
y = y_POT_targets(Index1);

%%%%%%%%%%%%%% approximation %%%%%%%%%%%%%%%
y_appr = -1+0.25*logsig(100*(x+1*0.125)) + 0.25*logsig(100*(x-1*0.125)) + 0.25*logsig(100*(x+0.375)) + 0.25*logsig(100*(x-0.375)) + ...
            0.5*logsig(100*(x+0.75)) +  0.5*logsig(100*(x-0.75));
%y_appr = -1 + 0.5 * (logsig(100 * (x + [0.25, 0.75])) + logsig(100 * (x - [0.25, 0.75])));

%y_appr = 0.1*sin(10*(x-0.1)) + x;
figure
plot(x, y,'LineWidth',5)
%legend('$Q_{uniform}$','Interpreter','latex') 
hold on 
plot(x, y_appr,'LineWidth',5)
Df = diff(y_appr);
%legend('Projection','Projection logsig approximation')

%legend(['$2^{\psi}$','Interpreter','latex', '$Q_{uniform}$','Interpreter','latex']) 
legend(['$Q_{uniform}$'],  ['$Q_{appr} = -1+0.5*\sum_{i=0}^{N-1} logsig(a_i \times (x+b_i \times 0.25 ))$'],'Interpreter','latex') 
set(gca,'FontSize',18, 'fontWeight','bold')
%%
% figure 
plot(x(1:length(x)-1), abs(Df),'LineWidth',3)
hold on
plot(x, 0.5*abs(y-y_appr),'LineWidth',3)
legend(['$\frac{\partial Q_{appr}}{\partial x}$'],  ['$abs(Q_{uniform} - Q_{appr})$'],'Interpreter','latex') 
plot(x, 0.5*abs(y-x),'LineWidth',3)
legend(['$\frac{\partial Q_{appr}}{\partial x}$'],  ['$abs(y - x)$'],'Interpreter','latex') 
%legend('diff Projection','diff Projection logsig approximation')
set(gca,'FontSize',18, 'fontWeight','bold')
