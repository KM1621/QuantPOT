%% 
x = 0:0.001:1;
v=x;
x=x';
% y_POT_targets = [-2^0 -2^-1 -2^-2 0 2^-2 2^-1 2^0];

p_0 = [0 2^0 2^-2];
p_1 = [0 2^-1 2^-3];
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
y = roundTargets(Index1);
y=y';
ft = fittype( 'piecewiseLine( x, a, b, c, d, k, jj, f, ll, f1, ll1, f2, ll2 )' )
f = fit( x, y, ft, 'StartPoint', [0, 0.05, 0.15, 0.175, 0.25, 0.275, 0.35, 0.375, 0.45, 0.5, 0.65, 0.75] )
figure
plot( f, x, y )
%% 