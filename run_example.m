%% run_example.m
% Minimal end-to-end example for the nested-LOSOCV hierarchical EEG-signature
% model. It trains the model on the shipped 10-subject example cohort, prints
% the internal leave-one-subject-out metrics, and saves a prediction scatter.
%
% NOTE: the 10-subject example is a de-identified SMOKE TEST so the pipeline
% runs in seconds. It is NOT expected to reproduce the paper's discovery-cohort
% correlation (r = 0.657), which requires the full N = 35 discovery cohort.
%
% Usage (from this folder, in MATLAB):
%   >> run_example
%
% Requirements: MATLAB with the Statistics and Machine Learning Toolbox
% (for `lasso`). Tested on R2022b.

clc; clear; close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, 'src'));

roi_mat  = fullfile(this_dir, 'example_data', 'roi_feature_cube_example.mat');
chan_mat = fullfile(this_dir, 'example_data', 'channel_feature_cube_example.mat');

out_dir = fullfile(this_dir, 'example_output');
if ~exist(out_dir,'dir'); mkdir(out_dir); end

%% ---- Train (nested LOSOCV) ----
cfg = struct();                 % use defaults (see train_hierarchical_model)
cfg.verbose = true;
OUT = train_hierarchical_model(roi_mat, chan_mat, cfg);

%% ---- Report ----
fprintf('\n=================== EXAMPLE RESULT (n=%d) ===================\n', numel(OUT.y_true));
fprintf('Nested-LOSO Pearson r = %.4f (perm p = %.4f)\n', OUT.r, OUT.perm_p);
fprintf('RMSE = %.3f | NRMSE = %.3f | MAE = %.3f | R^2 = %.3f\n', OUT.RMSE, OUT.NRMSE, OUT.MAE, OUT.R2);
fprintf('Final signature: band = %s | ROI = %s | lambda = %.4g | #channels = %d\n', ...
    OUT.final_model.band_name, OUT.final_model.roi_name, ...
    OUT.final_model.lambda, OUT.final_model.nnz);
fprintf('============================================================\n');

%% ---- Save tables and figure ----
writetable(OUT.per_fold,          fullfile(out_dir, 'example_per_fold_predictions.csv'));
writetable(OUT.channel_stability, fullfile(out_dir, 'example_channel_stability.csv'));

make_signature_scatter(OUT.pred_fused, OUT.y_true, ...
    fullfile(out_dir, 'example_prediction_scatter.png'), OUT.r, OUT.perm_p);

fprintf('Outputs written to: %s\n', out_dir);
