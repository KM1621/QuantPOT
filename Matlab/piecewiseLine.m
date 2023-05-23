function y = piecewiseLine(x,a,b,c,d,k,jj, f, ll, f1, ll1, f2, ll2)
% PIECEWISELINE   A line made of two pieces
% that is not continuous.

% y = zeros(size(x));
y = -1+a*logsig(100*(x+1*d)) + a*logsig(100*(x-1*d)) + b*logsig(100*(x+k)) + b*logsig(100*(x-k)) + ...
            c*logsig(100*(x+jj)) +  c*logsig(100*(x-jj)) + f*logsig(100*(x+ll)) +  f*logsig(100*(x-ll)) + ...
             f1*logsig(100*(x+ll1)) +  f1*logsig(100*(x-ll1)) + f2*logsig(100*(x+ll2)) +  f2*logsig(100*(x-ll2));
% y = -1+a*logsig(100*(x+1*d)) + a*logsig(100*(x-1*d)) + b*logsig(100*(x+k)) + b*logsig(100*(x-k)) + ...
%             c*logsig(100*(x+jj)) +  c*logsig(100*(x-jj)) + f*logsig(100*(x+ll)) +  f*logsig(100*(x-ll)) + ...
%              f1*logsig(100*(x+ll1)) +  f1*logsig(100*(x-ll1));
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