x = -1:0.001:1;
x=x';
y_POT_targets = [-2^0 -2^-1 -2^-2 0 2^-2 2^-1 2^0];
[~,Index1] = histc(v,[-Inf interp1(1:numel(y_POT_targets),y_POT_targets,0.5 + (1:numel(y_POT_targets)-1)) Inf]);
y = y_POT_targets(Index1);
y=y';
ft = fittype( 'piecewiseLine( x, a, b, c, d, k, jj )' )
f = fit( x, y, ft, 'StartPoint', [1, 0, 1, 0, 0.5, 1] )
plot( f, x, y )