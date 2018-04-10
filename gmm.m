function varargout = gmm(X, K_or_centroids)
    % input X:N-by-D data matrix
    % input K_or_centroids: K-by-D centroids
    
    % 阈值
    threshold = 1e-15;
    % 读取数据维度
    [N, D] = size(X);
    % 判断输入质心是否为标量
    if isscalar(K_or_centroids)
        % 是标量，随机选取K个质心
        K = K_or_centroids;
        rnpm = randperm(N); % 打乱的N个序列
        centroids = X(rnpm(1:K), :);
    else   % 矩阵，给出每一类的初始化
        K = size(K_or_centroids, 1);
        centroids = K_or_centroids;
    end
    
    % 定义模型初值
    [pMiu pPi pSigma] = init_params();
    
    Lprev = -inf;
    while true
        % E-step,估算出概率值
        % Px: N-by-K 
        Px = calc_prob();
        
        % pGamma新的值,样本点所占的权重
        % pPi:1-by-K     pGamma:N-by-K
        pGamma = Px ./ repmat(pPi, N, 1);
        % 对pGamma的每一行进行求和,sum(x,2):每一行求和
        pGamma = pGamma ./ repmat(sum(pGamma, 2) ,1 , K);
        
        % M-step
        % 每一个组件给予新的值
        Nk = sum(pGamma,1);
        pMiu = diag(1./Nk)*pGamma'*X;
        pPi = Nk/N;
        for kk = 1:K
           Xshift = X - repmat(pMiu(kk, :) ,N, 1);
           pSigma(:,:,kk) = (Xshift'*(diag(pGamma(:,kk))*Xshift)) / Nk(kk);
        end
        
        % 观察收敛，convergence
        L = sum(log(Px*pPi'));
        if L-Lprev < threshold
            break;
        end
        Lprev = L;
        
    end
    
    % 输出参数判定
    if nargout == 1
        varargout = {Px};
    else
        model = [];
        model.Miu = pMiu;
        model.Sigma = pSigma;
        model.Pi = pPi;
        varargout = {Px, model};
    end
    
    function [pMiu pPi pSigma] = init_params()
       pMiu = centroids; % 均值，K类的中心
       pPi = zeros(1, K); % 概率
       pSigma = zeros(D, D, K); % 协方差，每一个都是D-by-D
       
       % (X - pMiu)^2 = X^2 + pMiu^2 - 2*X*pMiu
       distmat = repmat(sum(X.*X, 2), 1, K) + repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*X*pMiu';
       [dummy labels] = min(distmat, [], 2); % 找出每一行的最小值,并标出列的位置
       
       for k=1:K   %初始化参数
           Xk = X(labels == k, :);
           pPi(k) = size(Xk, 1)/N;
           pSigma(:, :, k) = cov(Xk);
       end             
    end

    % 计算概率值
    function Px = calc_prob()
        Px = zeros(N,K);
        for k=1:K
            Xshift = X - repmat(pMiu(k,:),N,1);
            inv_pSigma = inv(pSigma(:,:,k)+diag(repmat(threshold, 1, size(pSigma(:,:,k),1))));
            tmp = sum((Xshift*inv_pSigma).*Xshift, 2);
            coef = (2*pi)^(-D/2)*sqrt(det(inv_pSigma));
            Px(:,k) = coef * exp(-1/2*tmp);
        end
    end
end