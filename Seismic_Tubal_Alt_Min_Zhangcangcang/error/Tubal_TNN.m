clc; 
clear all; 
close all;
% profile ong

%% data loading
% load('T_synthetic_tuabl_rank_3_32_32_300.mat');%T = T(11:21, 11:21, 81:200); %
load('T_synthetic_tuabl_rank_2.mat');% T = T(:, :, 1:256);   %加载我们自己合成的人工合成数据;
% load('T_synthetic_tuabl_rank_3.mat'); T = T(1:11, 1:11, 1:120);   %加载我们自己合成的人工合成数据;
% load('traces_100_100_1000.mat'); T = T(1:32, 1:32, 1:256);      %加载真实地震数据


% load('volume.mat');%真实地震数据
% T = volume(301:326,1:20,1:20);



szT = size(T);   
tubalRank = LowTubalCDF(T, 1);
r = tubalRank;

srs = [0.05 : 0.05 : 0.2];
sampling_rse = zeros(2, length(srs));
T1 = T;

for i = 1 : 10
for loop = 1 : length(srs)
    samplingrate = srs(loop);
    MatOmega1 = randsample([0 1],szT(3),true,[samplingrate, 1-samplingrate]);
    omega = ones(szT(1),szT(2),szT(3));
    for k=1:szT(3)
        if(MatOmega1(k)==0)
           omega(:,:,k)=zeros(szT(1),szT(2));
        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Tubal-Alt-Min                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RSE = zeros(1,2); 
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



   maxiter = 20;
   TubalAMCurve = zeros(maxiter, 1);
   iter=1;
   while iter <=maxiter
        [X_f_trans] = alter_min_LS_one_step(T_omega_f_trans, omega_f_trans * 1/k, Y_f_trans);
    
        for i =1:k
            X_f(:,:,i) = X_f_trans(:,:,i)';
        end
    
    %% Given X, do LS to get Y
        [Y_f] = alter_min_LS_one_step(T_omega_f, omega_f * 1/k, X_f);
    
        for i = 1: k
            Y_f_trans(:,:,i) = Y_f(:,:,i)'; 
        end
    %% The relative error:
       temp = 0;
       X_est = ifft(X_f, [], 3); 
       Y_est = ifft(Y_f, [], 3);
       T_est = tprod(X_est, Y_est);
       RSE(1) =  norm(T_est(:) - T1(:)) / norm(T1(:));
       TubalAMCurve(iter) = RSE(1);
       iter = iter + 1;
   end
   fprintf('***********************RSE = %d ***********\n',RSE(1));  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       TNN                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    normalize              =        max(T1(:))                     ;
    Xn                     =        T1/normalize                  ;
    [n1,n2,n3]             =        size(Xn);                      

    alpha                  =        1                             ;
    rho                    =        0.08                          ;
    myNorm                 =        'tSVD_1'                      ; % dont change for now

    A                      =        diag(sparse(double(omega(:)))); % sampling operator
    b                      =        A * Xn(:)                     ; % available data
    bb                     =        reshape(b,[n1,n2,n3]);

    %% ================ main process of completion =======================
    [tnnT, tnnCurve]   =    tensor_cpl_admm( A , b , Xn, rho , alpha , ...
                                            [n1,n2,n3] , 200, myNorm , 0 );
    tnnT               =        tnnT * normalize                 ;
    tnnT               =        reshape(tnnT,[n1,n2,n3])         ;
   TNNQ1 = -20*log10(norm(tnnT(:) - T1(:)) / norm(T1(:)));
   RSE(2) =  norm(tnnT(:) - T1(:)) / norm(T1(:));
   fprintf('***********************RSE = %d ***********\n',RSE(2)); 
   
   sampling_rse(:, loop) = sampling_rse(:, loop) + RSE(:);
end
end
sampling_rse = sampling_rse ./ 10;


% 绘制收敛速度
len = min([length(TubalAMCurve), length(tnnCurve)]);
figure;semilogy(TubalAMCurve(1:len), 'o-'); title(['Convergence Speed ']);
hold on; semilogy(tnnCurve(1:len), '*-');
legend('Tubal-Alt-Min', 'TNN'); 
xlabel('Iteration');ylabel('RSE in log-scale');
grid on;

% 重构误差
figure;semilogy([0.05 : 0.05 : 0.2]*100, sampling_rse(1,:), '+-'); title(['Reconstruction Error']);
hold on; semilogy([0.05 : 0.05 : 0.2]*100, sampling_rse(2, :), '*-');
legend( 'Tubal-Alt-Min','TNN'); 
xlabel('Slice Missing Rate %');ylabel('RSE in log-scale');
