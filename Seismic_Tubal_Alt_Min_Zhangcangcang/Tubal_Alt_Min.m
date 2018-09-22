clc; 
clear all; 
close all;
% profile on


%% data loading
% load('T_synthetic_tuabl_rank_3_32_32_300.mat');%T = T(11:21, 11:21, 81:200); %
load('T_synthetic_tuabl_rank_2.mat');T = T(:, :, 1:256);   %���������Լ��ϳɵ��˹��ϳ�����;
% load('T_synthetic_tuabl_rank_3.mat'); T = T(1:11, 1:11, 1:120);   %���������Լ��ϳɵ��˹��ϳ�����;
% load('traces_100_100_1000.mat'); T = T(1:32, 1:32, 1:256);      %������ʵ��������
% load('05HBC3D_ALL_POST_MIG_200_25_601_T.mat');T = T(:, :, 1:200);  %������ʵ��������


% m   = 60;    % the tensor is m * n * k
% n   = 60;
% k   = 20;
% r   = 3;        % the tubal-rank
% T = tprod(rand(m,r,k), rand(r,n,k));


szT = size(T);   
tubalRank = LowTubalCDF(T, 1);
% r = tubalRank;


%% transform dimension
T1 = permute(T,[3,1,2]);% ʱ��ά��Ϊ��һά������άΪcrossline��crosslineȱʧ��

%% tubalRank after transform dimension
tubalRank2 = LowTubalCDF(T1, 1);
% % %tubal-rankΪ7�����忴tSVD�ֽ��S�е�Ԫ����i=20��ʱ��Ż�С��0
r = tubalRank2;  
% r = 20;  

%% Slice sampling
szT1 = size(T1);
samplingrate = 0.01;
MatOmega = randsample([0 1],szT1(3),true,[samplingrate, 1-samplingrate]);
omega = ones(szT1(1),szT1(2),szT1(3));
for k=1:szT1(3)
    if(MatOmega(k)==0)
       omega(:,:,k)=zeros(szT1(1),szT1(2));
    end
end


%% 5-th frontal slice missing  
%���ջ�ͼ�������slice,�������ȱʧ��Ҫ��Ϊ�˻�ͼ����
omega(:,:,5)=zeros(szT1(1),szT1(2));


%% observations
[m,n,k] = size(T1);
T_f = fft(T1, [], 3);

T_omega = omega .* T1;  
T_omega_f = fft(T_omega,[],3);
omega_f = fft(omega, [], 3);

%% Alternating Minimization
%% X: m * r * k
%% Y: r * n * k
%% Given Y, do LS to get X
    Y = rand(r, n, k);
% Y = InitY(T_omega, r);
    Y_f = fft(Y, [], 3);

%% do the transpose for each frontal slice
    Y_f_trans = zeros(n,r,k);
    X_f = zeros(m,r,k);
    T_omega_f_trans = zeros(n,m,k);
    omega_f_trans = zeros(n,m,k);
for i = 1: k
     Y_f_trans(:,:,i) = Y_f(:,:,i)';
     T_omega_f_trans(:,:,i) = T_omega_f(:,:,i)';
     omega_f_trans(:,:,i) = omega_f(:,:,i)';
end

iter=1;
while iter <=15
    [X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans * 1/k, Y_f_trans);
    for i =1:k
        X_f(:,:,i) = X_f_trans(:,:,i)';
    end
    
    %% Given X, do LS to get Y
    [Y_f] = alter_min_LS_one_step(T_omega_f, omega_f * 1/k, X_f);
    
    for i = 1: k
        Y_f_trans(:,:,i) = Y_f(:,:,i)';
    end
    iter = iter + 1;
end

%% The relative error:
temp = 0;
X_est = ifft(X_f, [], 3); 
Y_est = ifft(Y_f, [], 3);
T_est = tprod(X_est, Y_est);
RSE =  norm(T_est(:) - T(:)) / norm(T(:));


%% figure �������slice��
figure;
subplot(1,3,1);
SeisPlot(T(:,5, :),{'figure', 'old'});
xlabel('CMP x number');ylabel('Time(ms)')
subplot(1,3,2);
SeisPlot(squeeze(T_omega(:,:, 5))',{'figure', 'old'});
xlabel('CMP x number');ylabel('Time(ms)') 
subplot(1,3,3);
SeisPlot(squeeze(T_est(:,:,5))',{'figure', 'old'});
xlabel('CMP x number');ylabel('Time(ms)')


