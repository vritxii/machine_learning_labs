function [C, I, iter] = Kmeans(X, K, maxIter, TOL, method)

% number of vectors in X
[vectors_num, dim] = size(X);

% compute a random permutation of all input vectors
R = randperm(vectors_num);

% construct indicator matrix (each entry corresponds to the cluster
% of each point in X)
I = zeros(vectors_num, 1);

% construct centers matrix
C = zeros(K, dim);

% take the first K points in the random permutation as the center sead
for k=1:K
    C(k,:) = X(R(k),:);
end

% iteration count
iter = 0;

% compute new clustering while the cumulative intracluster error in kept
% below the maximum allowed error, or the iterative process has not
% exceeded the maximum number of iterations permitted
while 1
    % find closest point
    for n=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = norm(X(n,:) - C(minIdx,:), 1);
        for j=1:K
            dist = norm(C(j,:) - X(n,:), 1);
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        
        % assign point to the closter center
        I(n) = minIdx;
    end
    
    % compute centers
    for k=1:K
        Nk = length(find(I == k));
        Pi = X(find(I == k), :);
        switch lower(method)
            case {'k_medoids','kmedoids'}
                 Dx2 = zeros(1,Nk);
                 for t=1:Nk
                    dx = Pi - ones(Nk,1)*Pi(t, :);
                    Dx2(t) = sum(sqrt(sum(dx.*dx,1)),2);
                 end
                 [~,min_ind] = min(Dx2);
                 C(k, :) = Pi(min_ind, :);
            otherwise
                C(k, :) = sum(X(find(I == k), :));
                C(k, :) = C(k, :) / Nk;
                method='k_means';
        end
    end
    
    % compute RSS error
    RSS_error = 0;
    for idx=1:vectors_num
        RSS_error = RSS_error + norm(X(idx, :) - C(I(idx),:), 2);
    end
    RSS_error = RSS_error / vectors_num;
    
    % increment iteration
    iter = iter + 1;
    
    % check stopping criteria
    if 1/RSS_error < TOL
        break;
    end
    
    if iter > maxIter
        iter = iter - 1;
        break;
    end
end

disp([method ' took ' int2str(iter) ' steps to converge']);
end