%% Plotting POT
% clear all
clc
close all

x = -1:0.01:1;
y_binary = sign(x);
b=2;
step_size = 2/(2^b);
y_1 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*1)/1));
b=4;
step_size = 2/(2^b);
y_2 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*2)/2));
%y_3 = round(x*8)/8;
figure

box off
plot(x, x,'LineWidth',2)
hold on
plot(x, y_binary,'LineWidth',2, 'LineStyle',':')
hold on
plot(x, y_1,'LineWidth',2, 'LineStyle','--')
hold on
plot(x, y_2,'LineWidth',2, 'LineStyle','-.')
%hold on
%plot(x, y_3,'LineWidth',2, 'LineStyle',':')
axis([-1.2 1.2 -1.2 1.2])
xlabel('Input')
ylabel('Projection')
legend('STE','Binary', '2-bit', '4-bit')
set(gca,'FontSize',18, 'fontWeight','bold')


%% Plotting uniform and non-uniform quantization
clc
close all

x = 0:0.01:1;
v=x;
y_binary = sign(x);
b=3;
step_size = 2/(2^b);
y_3 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*1)/1));
% y_POT = sign(x)*2.^log2(abs(round(x*2)/2));
y_POT_targets = [0 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2  2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_POT_targets),y_POT_targets,0.5 + (1:numel(y_POT_targets)-1)) Inf]);
y_POT = y_POT_targets(Index1);

%APOT
%roundTargets = [0 sort(10*rand(1,10000)) 10];
p_0 = [0 2^0 2^-2 2^-4];
p_1 = [0 2^-1 2^-3 2^-5];
p_target = zeros(1, length(p_0)*length(p_1));
kk=1;
for i = 1 : length(p_0)
    for j = 1 : length(p_1)
        p_target(1, kk) = p_0(1, i) + p_1(1, j);
        kk=kk+1;        
    end
end
roundTargets = sort(p_target);

[~,Index1] = histc(v,[-Inf interp1(1:numel(roundTargets),roundTargets,0.5 + (1:numel(roundTargets)-1)) Inf]);
y_APOT = roundTargets(Index1);

box off
subplot(3,1,1);
plot(x, y_3,'LineWidth',2, 'LineStyle',':')
set(gca,'fontsize',15, 'fontweight','bold')
title('Uniform: 2 bits')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 20, 'fontweight','bold')
xlabel({'x'})
hold on
subplot(3,1,2);
plot(x, y_POT,'LineWidth',2, 'LineStyle',':')
set(gca,'fontsize',15, 'fontweight','bold')
title('POT (Non uniform): 3  bits')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 20, 'fontweight','bold')
xlabel({'x'})
hold on
subplot(3,1,3);
plot(x, y_APOT,'LineWidth',2, 'LineStyle',':')
set(gca,'fontsize',15, 'fontweight','bold')
title('APOT (Non uniform): n = 2 b = 2; p_0 = [0 2^0 2^{-2} 2^{-4}] and p_1 = [0 2^{-1} 2^{-3} 2^{-5}]')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 20, 'fontweight','bold')
xlabel({'x'})


% xlabel('input')
% ylabel('projection')
%legend('Uniform','POT', 'APOT')

%% Plotting POT quantization with different bitwidth
clc
close all

x = 0:0.0001:1;
v=x;
y_binary = sign(x);

%b=2; 
%step_size = 2/(2^b);
%y_2 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*1)/1));
y_2_targets = [2^-3 2^-2  2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_2_targets),y_2_targets,0.5 + (1:numel(y_2_targets)-1)) Inf]);
y_2 = y_2_targets(Index1);

%b=3; 
%step_size = 2/(2^b);
%y_3 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*1)/1));
y_3_targets = [0 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2  2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_3_targets),y_3_targets,0.5 + (1:numel(y_3_targets)-1)) Inf]);
y_3 = y_3_targets(Index1);

%b=4; 
%step_size = 2/(2^b);
%y_4 = round(x/step_size)*step_size;  %sign(x).*2.^log2(abs(round(x*1)/1));
% y_POT = sign(x)*2.^log2(abs(round(x*2)/2));
y_4_targets = [0 2^-14 2^-13 2^-12 2^-11 2^-10 2^-9  2^-8 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2  2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_4_targets),y_4_targets,0.5 + (1:numel(y_4_targets)-1)) Inf]);
y_4 = y_4_targets(Index1);


box off
subplot(3,1,1);
plot(x, y_2,'LineWidth',2)
set(gca,'fontsize',15, 'fontweight','bold')
title('Bits = 2')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 24, 'fontweight','bold')
xlabel({'x'})
hold on
subplot(3,1,2);
plot(x, y_3,'LineWidth',2)
set(gca,'fontsize',15, 'fontweight','bold')
title('Bits = 3')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 24, 'fontweight','bold')
xlabel({'x'})
hold on
subplot(3,1,3);
plot(x, y_4,'LineWidth',2)
set(gca,'fontsize',15, 'fontweight','bold')
title('Bits = 4')
ylabel({'$f_{proj}$'},'Interpreter','latex', 'FontSize', 24, 'fontweight','bold')
xlabel({'x'})
%% Scratch pad
% roundTargets = [0 2.7 8 11.1];
% v = [0.1 2.2 1.6 7.3 10];
v = 10*rand(1,1000000);
roundTargets = [0 sort(10*rand(1,10000)) 10];

[~,Index1] = histc(v,[-Inf interp1(1:numel(roundTargets),roundTargets,0.5 + (1:numel(roundTargets)-1)) Inf]);
vRounded1 = roundTargets(Index1);
