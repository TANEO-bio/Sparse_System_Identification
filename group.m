clear all;
% Sparse system identification with different algorithms
% Please acknowledge the paper
% R. C. de Lamare and R. Sampaio-Neto, "Sparsity-Aware 
% Adaptive Algorithms Based on Lassoernating Optimization 
% and Shrinkage", IEEE Signal Processing Letters, vol. 21, 
% no. 2, February 2014.
% 
I = 10; %			-	number of repetitions
K = 200; %			-	iterations
N = 32; %           -   length of the system
S = 16;  %           -   number of nonzero coefficients
sigmax = 1; %       -	standard deviation of the input signal
c = ones(S,1);
c = c/sqrt(c'*c); % generation of S complex coefficients
ind_r = round(1 + (N-1).*rand(S,1)); % -	sample index
Wo  = zeros(N,1);
Wo(ind_r)  = c;
Wo = Wo/sqrt(Wo'*Wo)/2; %      -	plant/system to be identified
sigman = sqrt(0.01); %    -	standard deviation of the noise
mu = 0.25;  %		-	step size
mu_lms = 0.1;
mu_or = 0.1;
mu_m = 0.1;
mu_a = 0.1;
ep = 10.0; %
reg = 0.1;
gamma = 0.0002; %
gamma_g = 0.0004;
gamma_a = 0.0001;
lambda_=0.02;
lambda_gS=0.005;
lambda_g = 0.0001;

group_id = sign(Wo);
group_id = (group_id == 0:length(unique(group_id)) -1) * 1;
% MSE vectors 
MSE_lms = zeros(K,1);
MSE_Lasso = zeros(K,1);
MSE_gLasso = zeros(K,1);
MSE_SCAD = zeros(K,1);
MSE_gSCAD = zeros(K,1);
MSE_or = zeros(K,1);
% Parameters of the oracle algorithm
p = zeros(N,1);
p(ind_r) = 1;
P = diag(p);
% MSD vectors
MSD_lms = zeros(K,1);
MSD_or = zeros(K,1);
MSD_Lasso = zeros(K,1);
MSD_gLasso = zeros(K,1);
MSD_SCAD = zeros(K,1);
MSD_gSCAD = zeros(K,1);
% Minimum MSE vector
MSEmin=zeros(K,1);
rho = 0.005;
Q = 0;
for i=1:I	
   t = randn(N,1)*sigmax;  
   y = zeros(N,1);
   y(ind_r) = t(ind_r);  	
   w_lms = zeros(N,1);
   w_Lasso = zeros(N,1);
   w_gLasso = zeros(N,1);
   w_SCAD = zeros(N,1);
   w_gSCAD = zeros(N,1);
   a_Lasso = 1*ones(N,1);
   a_gLasso = 1*ones(N,1);
   a_SCAD = 1*ones(N,1);
   a_gSCAD = 1*ones(N,1);
   w_or = zeros(N,1);
   % input
   M = 16;
   data = randi([0 M-1],1000,1); % Random message
   x = qammod(data,M,'UnitAveragePower',true);	
   x = x(1:K);
   
   % noise
   n = randn(K,1)*sigman;
   d_list = zeros(K, 1);
   d_hat_lms_list = zeros(K, 1);
   d_hat_Lasso_list = zeros(K, 1);
   d_hat_gLasso_list = zeros(K, 1);
   d_hat_SCAD_list = zeros(K, 1);
   d_hat_gSCAD_list = zeros(K, 1);
   d_hat_or_list = zeros(K, 1);
   w_hat_lms_list = zeros(K, N);
   w_hat_Lasso_list = zeros(K, N);
   w_hat_gLasso_list = zeros(K, N);
   w_hat_SCAD_list = zeros(K, N);
   w_hat_gSCAD_list = zeros(K, N);
   for k = 1:K
      y = [x(k) 
          y(1:N-1) ];
      % desired signal 
      d = Wo'*y;
      d_list(k) = d;
      % output estimates 
      z_lms = w_lms'*y;
      z_Lasso = w_Lasso'*diag(a_Lasso)*y;
      z_gLasso = w_gLasso'*diag(a_gLasso)*y;
      z_SCAD = w_SCAD'*diag(a_SCAD)*y;
      z_gSCAD = w_gSCAD'*diag(a_gSCAD)*y;
      z_or = w_or'*P*y;
      d_hat_lms_list(k) = z_lms;
      d_hat_Lasso_list(k) = z_Lasso;
      d_hat_gLasso_list(k) = z_gLasso;
      d_hat_SCAD_list(k) = z_SCAD;
      d_hat_gSCAD_list(k) = z_gSCAD;
      d_hat_or_list(k) = z_or;
      % error signal
      e_lms = d+n(k)- z_lms;	  	
      e_Lasso = d + n(k) - z_Lasso;
      e_gLasso = d + n(k) - z_gLasso;
      e_SCAD = d + n(k) - z_SCAD;
      e_gSCAD = d + n(k) - z_gSCAD;
      e_or = d + n(k) - z_or;
      % new/updated filters
      a_Lasso = a_Lasso + mu_a/(reg + y'*diag(w_Lasso')*diag(w_Lasso)*y)*e_Lasso*diag(w_Lasso)*conj(y);
      a_gLasso = a_gLasso + mu_a/(reg + y'*diag(w_gLasso')*diag(w_gLasso)*y)*e_gLasso*diag(w_gLasso)*conj(y);
      a_SCAD = a_SCAD + mu_a/(reg + y'*diag(w_SCAD')*diag(w_SCAD)*y)*e_SCAD*diag(w_SCAD)*conj(y);
      a_gSCAD = a_gSCAD + mu_a/(reg + y'*diag(w_gSCAD')*diag(w_gSCAD)*y)*e_gSCAD*diag(w_gSCAD)*conj(y);
      e_Lasso = d + n(k) - w_Lasso'*diag(a_Lasso)*y;
      e_gLasso = d + n(k) - w_gLasso'*diag(a_gLasso)*y;
      e_SCAD = d + n(k) - w_SCAD'*diag(a_SCAD)*y;
      e_gSCAD = d + n(k) - w_gSCAD'*diag(a_gSCAD)*y;
      w_lms = w_lms + mu_lms/(reg + y'*y)*conj(e_lms)*y; %/(reg + y'*y)
      w_Lasso = w_Lasso  + mu/(reg + y'*diag(a_Lasso)*diag(a_Lasso')*y)*conj(e_Lasso)*diag(a_Lasso)*y - gamma.*(sign(real(w_Lasso)) + 1i*sign(imag(w_Lasso)))./(1+ep.*abs(w_Lasso));
      w_gLasso = w_gLasso  + mu/(reg + y'*diag(a_gLasso)*diag(a_gLasso')*y)*conj(e_gLasso)*diag(a_gLasso)*y - gamma_g .* w_gLasso ./ (group_id * sqrt(group_id' * (w_gLasso.^2))+ 0.00000001) ./(1+ep.*abs(w_gLasso));
      w_SCAD = w_SCAD  + mu/(reg + y'*diag(a_SCAD)*diag(a_SCAD')*y)*conj(e_SCAD)*diag(a_SCAD)*y - gamma .*(sign(max(-1 * abs(real(w_SCAD)) + lambda_, 0)) .* sign(real(w_SCAD)) + (3.7 .* lambda_ - real(abs(w_SCAD))) / (2.7 .* lambda_) .* sign(max(abs(real(w_SCAD)) - lambda_, 0)) .* sign(real(w_SCAD)) + sign(max(-1 * abs(imag(w_SCAD)) + lambda_, 0)) .* sign(imag(w_SCAD)) + (3.7 .* lambda_ - abs(imag(w_SCAD))) / (2.7 .* lambda_) .* sign(max(abs(imag(w_SCAD)) - lambda_, 0)) .* sign(imag(w_SCAD)))./(1+ep.*abs(w_SCAD));
      w_gSCAD = w_gSCAD  + mu/(reg + y'*diag(a_gSCAD)*diag(a_gSCAD')*y)*conj(e_gSCAD)*diag(a_gSCAD)*y - gamma .*(sign(max(-1 * abs(real(w_gSCAD)) + lambda_gS, 0)) .* w_gSCAD ./ (group_id * sqrt(group_id' * (w_gSCAD.^2))+ 0.00000001) + (3.7 .* lambda_gS - real(abs(w_gSCAD))) / (2.7 .* lambda_gS) .* sign(max(abs(real(w_gSCAD)) - lambda_gS, 0)) .* w_gSCAD ./ (group_id * sqrt(group_id' * (w_gSCAD.^2))+ 0.00000001) + sign(max(-1 * abs(imag(w_gSCAD)) + lambda_gS, 0)) .* w_gSCAD ./ (group_id * sqrt(group_id' * (w_gSCAD.^2))+ 0.00000001) + (3.7 .* lambda_gS - abs(imag(w_gSCAD))) / (2.7 .* lambda_gS) .* sign(max(abs(imag(w_gSCAD)) - lambda_gS, 0)) .* w_gSCAD ./ (group_id * sqrt(group_id' * (w_gSCAD.^2))+ 0.00000001))./(1+ep.*abs(w_gSCAD));
      w_or = w_or + mu_or/(reg + y'*P*y)*conj(e_or)*P*y;
      if k > Q
          w_Lasso(abs(w_Lasso) < rho) = 0;
          w_gLasso(abs(w_gLasso) < rho) = 0;
          w_SCAD(abs(w_SCAD) < rho) =0;
          w_gSCAD(abs(w_gSCAD) < rho) =0;
          a_Lasso(abs(a_Lasso) < rho) = 0;
          a_gLasso(abs(a_gLasso) < rho) = 0;
          a_SCAD(abs(a_SCAD) < rho) =0;
          a_gSCAD(abs(a_gSCAD) < rho) =0;
      end
      w_hat_lms_list(k, :) = w_lms;
      w_hat_Lasso_list(k, :) = w_Lasso;
      w_hat_gLasso_list(k, :) = w_gLasso;
      w_hat_SCAD_list(k, :) = w_SCAD;
      w_hat_gSCAD_list(k, :) = w_gSCAD;
      w_hat_or_list(k, :) = w_or;
      % accummulation of MSE
      MSE_lms(k) = MSE_lms(k)+abs(e_lms)^2;	
      MSE_Lasso(k) = MSE_Lasso(k)+abs(e_Lasso)^2;
      MSE_gLasso(k) = MSE_gLasso(k)+abs(e_gLasso)^2;
      MSE_SCAD(k) = MSE_SCAD(k)+abs(e_SCAD)^2;
      MSE_gSCAD(k) = MSE_gSCAD(k)+abs(e_gSCAD)^2;
      MSE_or(k) = MSE_or(k)+abs(e_or)^2;
      % accummulation of MSE
      MSEmin(k) = MSEmin(k)+abs(n(k))^2;
      % accummulation of MSD
      MSD_lms(k) = MSD_lms(k)+(w_lms-Wo)'*(w_lms-Wo);
      MSD_Lasso(k) = MSD_Lasso(k)+(diag(a_Lasso')*w_Lasso-Wo)'*(diag(a_Lasso')*w_Lasso-Wo);
      MSD_gLasso(k) = MSD_gLasso(k)+(diag(a_gLasso')*w_gLasso-Wo)'*(diag(a_gLasso')*w_gLasso-Wo);
      MSD_SCAD(k) = MSD_SCAD(k)+(diag(a_SCAD')*w_SCAD-Wo)'*(diag(a_SCAD')*w_SCAD-Wo);
      MSD_gSCAD(k) = MSD_gSCAD(k)+(diag(a_gSCAD')*w_gSCAD-Wo)'*(diag(a_gSCAD')*w_gSCAD-Wo);
      MSD_or(k) = MSD_or(k)+(w_or-Wo)'*(w_or-Wo);
   end
end
% sample index
ind=1:(K);
MSE_lms = MSE_lms/I;
MSE_Lasso = MSE_Lasso/I;
MSE_gLasso = MSE_gLasso/I;
MSE_SCAD = MSE_SCAD/I;
MSE_gSCAD = MSE_gSCAD/I;
MSE_or = MSE_or/I;
MSEmin = MSEmin/I;
% Misadjustment computation 
M=MSE_lms./MSEmin-1; 
scal = [1 50:50:K];
ind = ind(scal);

plot(1:K,log10(MSE_lms), 1:K,log10(MSE_Lasso), 1:K,log10(MSE_SCAD), 1:K,log10(MSE_gLasso), 1:K,log10(MSE_gSCAD));
xlabel('iterations');
ylabel('log10(MSE)');
legend('LMS','Lasso-LMS','SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS');
title('MSE of predicted signal');
saveas(gcf,'mse.png');

plot(1:K,log10(MSD_lms), 1:K,log10(MSD_Lasso), 1:K,log10(MSD_SCAD), 1:K,log10(MSD_gLasso), 1:K,log10(MSD_gSCAD));
xlabel('iterations');
ylabel('log10(MSD)');
legend('LMS','Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS');
title('MSD of predicted signal');
saveas(gcf,'msd.png');

plot(1:K, real(d_hat_lms_list(1:K)), 1:K, real(d_hat_Lasso_list(1:K)), 1:K, real(d_hat_SCAD_list(1:K)), 1:K, real(d_hat_gLasso_list(1:K)), 1:K, real(d_hat_gSCAD_list(1:K)), 1:K, real(d_list(1:K)), "r");
xlabel('iterations');
ylabel('Amplitude');
legend('LMS', 'Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS', 'True');
title('Predicted signal');
saveas(gcf,'d_hat_real.png');

plot(1:K, real(d_hat_lms_list(1:K)), 1:K, imag(d_hat_Lasso_list(1:K)), 1:K, imag(d_hat_SCAD_list(1:K)), 1:K, imag(d_hat_gLasso_list(1:K)), 1:K, imag(d_hat_gSCAD_list(1:K)), 1:K, imag(d_list(1:K)), "r");
xlabel('iterations');
ylabel('Amplitude');
legend('LMS', 'Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS', 'True');
title('Predicted signal');
saveas(gcf,'d_hat_imag.png');

plot(1:32, real(w_lms), 1:32, real(w_Lasso), 1:32, real(w_SCAD), 1:32, real(w_gLasso), 1:32, real(w_gSCAD));
xlabel('index');
ylabel('w');
legend('LMS', 'Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS');
title('Predicted filter coefficients');
saveas(gcf,'coef.png');

plot(real(x));
title('Re(Input signal)');
saveas(gcf,'x_real.png');

plot(imag(x));
title('Im(Input signal)');
saveas(gcf,'x_imag.png');

plot(real(d_list));
title('Re(Desired signal)');
saveas(gcf,'d_real.png');

plot(imag(d_list));
title('Im(Desired signal)');
saveas(gcf,'d_imag.png');

plot(1:K, real(w_hat_lms_list(:, ind_r(1))), 1:K, real(w_hat_Lasso_list(:, ind_r(1))), 1:K, real(w_hat_SCAD_list(:, ind_r(1))), 1:K, real(w_hat_gLasso_list(:, ind_r(1))), 1:K, real(w_hat_gSCAD_list(:, ind_r(1))), 1:K, real(Wo(ind_r(1)) * ones(K, 1)), "r");
xlabel('iterations');
ylabel('W(t)');
legend('LMS','Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS', 'True');
title('Time variation of non-zero filter coefficient');
saveas(gcf,'W_nonzero.png');
ind_zero = 1:N;
ind_zero(ind_r) = [];
plot(1:K, real(w_hat_lms_list(:, ind_zero(1))), 1:K, real(w_hat_Lasso_list(:, ind_zero(1))), 1:K, real(w_hat_SCAD_list(:, ind_zero(1))), 1:K, real(w_hat_gLasso_list(:, ind_zero(1))), 1:K, real(w_hat_gSCAD_list(:, ind_zero(1))), 1:K, real(Wo(ind_zero(1)) * ones(K, 1)), "r");
xlabel('iterations');
ylabel('W(t)');
legend('LMS','Lasso-LMS', 'SCAD-LMS', 'GroupLasso-LMS', 'GroupSCAD-LMS', 'True');
title('Time variation of zero filter coefficient');
saveas(gcf,'W_zeros.png');