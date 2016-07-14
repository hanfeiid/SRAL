%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file:     nordland_pr.m
% Author:   Fei Han
% Email:    fhan@mines.edu
% Date:     05/07/2016
% Experiment of the RSSW16 paper
% Life-Long Place Recognition by Shared Representative Appearance Learning
% Result:     
%           weight	Features
%           ----------------------
%           0.0442  color feature
%           0.0002  GIST feature
%           1.7188  HOG feature     * Most representive feature
%           0.0113  LBP feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%% load feature vectors of each modality using the Nordland dataset
load colorfeature.mat;
load GISTfeature.mat;
load HOGfeature.mat;
load LBPfeature.mat;

%% initialization
classIndicator = eye(4);    % 4 classes: spring, summer, autumn, winter

X = feature;                % X matrix in Algorithm 1
Y = [repmat(classIndicator(1,:),9000,1);
    repmat(classIndicator(2,:),9000,1);
    repmat(classIndicator(3,:),9000,1);
    repmat(classIndicator(4,:),9000,1);];   % Y matrix
XX = X * X';
XY = X * Y;

W = zeros(size(X,1),size(Y,2)); % weight matrix W

ITER = 50;                  % maximum iteration
J = zeros(ITER, 1);         % Objective value in Algorithm 1
Icolor = ones(dColor,1);    % color feature modality dimension
Igist = ones(dGist,1);      % gist feature modality dimension
Ihog = ones(dHog,1);        % hog feature modality dimension
Ilbp = ones(dLbp,1);        % lbp feature modality dimension

%% realization of Algorithm 1: iterations to obtain optimal W
for iteration = 1 : ITER    
    % l-21 norm
    W_21 = sqrt( sum(W.^2,2) + eps);
    D_21 = diag(1./W_21);
    
    % group norm
    vColor = Icolor / (norm(W(1:dColor,:),'fro') + eps);
    vGist = Igist / (norm(W(dColor+1:dColor+dGist,:),'fro') + eps);
    vHog = Ihog / (norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro') + eps);
    vLbp = Ilbp / (norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro') + eps);
    D_G = diag([vColor; vGist; vHog; vLbp]);
    
    % update W
    for i = 1 : 4
        W(:,i) = (XX + lambda_G * D_G + lambda_21 * D_21) \ XY(:,i);
    end
    J_G = norm(W(1:dColor,:),'fro') + norm(W(dColor+1:dColor+dGist,:),'fro') ...
            + norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro') ...
            + norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro') + 4 * eps;
    J_21 = sum (sqrt(sum(W.*W,2)+eps));
    J(iteration) = 0.5 * norm(X'*W-Y, 'fro')^2 + lambda_G * J_G + lambda_21 * J_21;
    display([iteration J(iteration)]);
    
    if (iteration > 1 && abs(J(iteration) - J(iteration-1)) <= 0.0001)
            break;
    end
end

% output
save('norway_W.mat', 'W');                  % W matrix
display([norm(W(1:dColor,:),'fro'); ...     % display weight for each feature
 norm(W(dColor+1:dColor+dGist,:),'fro'); ...
 norm(W(dColor+dGist+1:dColor+dGist+dHog,:),'fro'); ...
 norm(W(dColor+dGist+dHog+1:dColor+dGist+dHog+dLbp,:),'fro')]);