function y = piecewiseLine(x,a,b,c,d,k,jj)
% PIECEWISELINE   A line made of two pieces
% that is not continuous.

% y = zeros(size(x));
y = -1+a*logsig(100*(x+1*d)) + a*logsig(100*(x-1*d)) + b*logsig(100*(x+k)) + b*logsig(100*(x-k)) + ...
            c*logsig(100*(x+jj)) +  c*logsig(100*(x-jj));
% This example includes a for-loop and if statement
% purely for example purposes.
% for i = 1:length(x)
%     if x(i) < k
%         y(i) = a + b.* x(i);
%     else
%         y(i) = c + d.* x(i);
%     end
% end
end