% File:     nordland_pr.m
% Author:   Fei Han
% Email:    fhan@mines.edu
% Date:     05/07/2016
% For:      Experiment using Nordland dataset for RSS16 Workshop paper
%           https://inside.mines.edu/~fhan/publication/pdf/rss16_roms.pdf

clear all;
close all;
clc;

%% load features
% Each image is represented by four different feature modalities.
load colorfeature.mat;
load GISTfeature.mat;
load HOGfeature.mat;
load LBPfeature.mat;

%% initialization

% X matrix in Algorithm 1
X = [I1color I2color I3color I4color;
     I1gist  I2gist  I3gist  I4gist;
     I1hog   I2hog   I3hog   I4hog;
     I1lbp   I2lbp   I3lbp   I4lbp];
 
% dimension of each feature modality
dColor = size(I1color, 1);
dGist = size(I1gist, 1);
dHog = size(I1hog, 1);
dLbp = size(I1lbp, 1);

[d, n] = size(X);

% Y matrix in Algorithm 1
class = eye(4);    % 4 classes: spring, summer, autumn, winter
Y = [repmat(class(1,:),n/4,1);
    repmat(class(2,:),n/4,1);
    repmat(class(3,:),n/4,1);
    repmat(class(4,:),n/4,1)];

lambda_M = 0.1;                 % lambda for l-M norm 
lambda_21 = 0.1;                % lambda for l-2,1 norm
W = zeros(size(X,1),size(Y,2)); % weight matrix W

XX = X * X';
XY = X * Y;

ITER = 50;                      % maximum iteration times
obj = zeros(ITER, 1);           % Objective value in Algorithm 1
Icolor = ones(dColor,1);        % color feature modality
Igist = ones(dGist,1);          % gist feature modality
Ihog = ones(dHog,1);            % hog feature modality
Ilbp = ones(dLbp,1);            % lbp feature modality

%% realization of Algorithm 1: iterations to obtain optimal W
for iter_idx = 1 : ITER    
    % l-2,1 norm
    W_21 = sqrt( sum(W.^2,2) + eps);
    D_21 = diag(1./W_21);
    
    % M-norm
    vColor = Icolor / (norm(W(1:dColor,:),'fro') + eps);
    vGist = Igist / (norm(W(dColor+1:dColor+dGist,:),'fro') + eps);
    vHog = Ihog / (norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro') + eps);
    vLbp = Ilbp / (norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro') + eps);
    D_M = diag([vColor; vGist; vHog; vLbp]);
    
    % update W
    for i = 1 : 4
        W(:,i) = (XX + lambda_M * D_M + lambda_21 * D_21) \ XY(:,i);
    end
    obj_M = norm(W(1:dColor,:),'fro') + norm(W(dColor+1:dColor+dGist,:),'fro') ...
            + norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro') ...
            + norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro') + 4 * eps;
    obj_21 = sum(sqrt(sum(W.*W,2)+eps));
    obj(iter_idx) = 0.5 * norm(X'*W-Y,'fro')^2 + lambda_M * obj_M + lambda_21 * obj_21;
    display(sprintf('iteration %d: objective value = %f',iter_idx, obj(iter_idx)));
    
    % end condition
    if (iter_idx > 1 && abs(obj(iter_idx) - obj(iter_idx-1)) <= 0.0001)
    	break;
    end
end

% output
save('nordland_W.mat', 'W');                  % W matrix
display([norm(W(1:dColor,:),'fro'); ...     % display weight for each feature
 norm(W(dColor+1:dColor+dGist,:),'fro'); ...
 norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro'); ...
 norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro')]);
