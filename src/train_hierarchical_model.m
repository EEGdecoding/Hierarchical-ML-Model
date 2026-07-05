function OUT = train_hierarchical_model(roi_mat_path, chan_mat_path, cfg)
% TRAIN_HIERARCHICAL_MODEL  Nested-LOSOCV hierarchical EEG-signature model.
%
%   Implements the discovery-cohort model of:
%   "A Machine-Learning-Based Electroencephalographic Signature Predicts and
%    Tracks Response to Accelerated Continuous Theta-Burst Stimulation in
%    Adolescent Depression".
%
%   The model predicts the percentage improvement in HAMD (pctHAMD) from
%   resting-state EEG features using a two-stage hierarchy, with all
%   hyper-parameters chosen inside a nested leave-one-subject-out
%   cross-validation (LOSOCV):
%
%     OUTER loop  (performance assessment)
%        leave one subject out -> produce one honest out-of-fold prediction
%        per subject.
%
%     INNER loop  (hyper-parameter selection, run only on the outer-training
%                  subjects)
%        Band level   : the frequency band is treated as the top-level
%                        hyper-parameter and chosen by the fused inner-LOSO MSE.
%        Stage 1 (ROI level)     : for each candidate band, a single best ROI
%                        is chosen by inner-LOSO ROI-only MSE, then a univariate
%                        linear regression  yhat_roi = b0 + b1 * ROI  is fit.
%        Stage 2 (Channel level) : the ROI residual  e = y - yhat_roi  is fit
%                        with a LASSO over the channel features; lambda is
%                        chosen by fused inner-LOSO MSE.
%        Fused prediction        :  yhat = yhat_roi + ehat.
%
%   After the outer loop, the same selection procedure is re-run on the FULL
%   cohort to give the final signature (selected band, ROI, univariate ROI
%   regression and channel weights).
%
%   USAGE
%     OUT = train_hierarchical_model(roi_mat_path, chan_mat_path)
%     OUT = train_hierarchical_model(roi_mat_path, chan_mat_path, cfg)
%
%   INPUTS
%     roi_mat_path   path to a MAT file with a struct 'roi_data' holding:
%                      X_roi      [nSub x nROI  x nFeat]
%                      subjIDs    [nSub x 1] string/char
%                      roiNames   [nROI x 1]  string/char
%                      featNames  [nFeat x 1] string/char  (e.g. Rel_Alpha)
%                      pctHAMD    [nSub x 1]  outcome (% HAMD reduction)
%     chan_mat_path  path to a MAT file with a struct 'channel_data' holding:
%                      X_chan     [nSub x nChan x nFeat]
%                      subjIDs, chanNames, featNames, pctHAMD  (as above)
%     cfg            optional struct, see DEFAULT_CFG below.
%
%   OUTPUT (struct OUT), main fields
%     subjIDs, y_true
%     pred_fused        out-of-fold fused prediction per subject
%     pred_roi, pred_res
%     r, perm_p, RMSE, NRMSE, MAE, R2      internal nested-LOSO metrics
%     per_fold                             table of per-fold band/ROI/lambda
%     channel_stability                    channel selection frequency table
%     final_model                          band/ROI/channel weights refit on all
%
%   See also LOAD_ROI_CUBE, LOAD_CHANNEL_CUBE, MAKE_SIGNATURE_SCATTER.

if nargin < 3 || isempty(cfg); cfg = struct(); end
cfg = fill_default_cfg(cfg);
rng(cfg.rng_seed, 'twister');

%% ---------- Load & align the two feature cubes ----------
[X_roi_raw, subjIDs_roi_raw, roiNames, roi_featNames, y_roi] = load_roi_cube(roi_mat_path);
[X_chan_raw, subjIDs_chan_raw, chanNames, chan_featNames, y_chan] = load_channel_cube(chan_mat_path);

[nSub_roi,~,~]  = size(X_roi_raw);
[nSub_chan,~,~] = size(X_chan_raw);
[subjIDs_roi,  subjKey_roi]  = normalize_subject_ids(subjIDs_roi_raw,  nSub_roi,  "SUB");
[subjIDs_chan, subjKey_chan] = normalize_subject_ids(subjIDs_chan_raw, nSub_chan, "SUB");

[~, ia_roi, ia_chan] = intersect(subjKey_roi, subjKey_chan, 'stable');
if isempty(ia_roi)
    error('ROI and channel cubes share no common subject IDs.');
end
X_roi   = X_roi_raw(ia_roi,:,:);
X_chan  = X_chan_raw(ia_chan,:,:);
y       = y_roi(ia_roi);
subjIDs = subjIDs_roi(ia_roi);
y_chan_al = y_chan(ia_chan);
if any(abs(y - y_chan_al) > 1e-8 & isfinite(y) & isfinite(y_chan_al))
    warning('ROI and channel outcomes differ; using the ROI pctHAMD.');
end

roiNames  = string(roiNames(:));
chanNames = string(chanNames(:));
nSub  = numel(y);
nROI  = size(X_roi,2);
nChan = size(X_chan,2);

% map requested bands to the correct feature slice in each cube
[roi_feat_idx, chan_feat_idx] = build_band_index(cfg.target_feats, roi_featNames, chan_featNames);
nBand = numel(cfg.target_feats);

if cfg.verbose
    fprintf('Cohort: %d subjects | %d ROIs | %d channels | %d bands\n', ...
        nSub, nROI, nChan, nBand);
end

%% ---------- OUTER loop: nested-LOSO out-of-fold prediction ----------
pred_fused = nan(nSub,1);
pred_roi   = nan(nSub,1);
pred_res   = nan(nSub,1);
best_band  = nan(nSub,1);
best_roi   = nan(nSub,1);
best_lambda= nan(nSub,1);
nnz_chan   = nan(nSub,1);
beta_cell  = cell(nSub,1);
chan_mask  = false(nSub, nChan);

for i = 1:nSub
    tr = true(nSub,1); tr(i) = false;

    fold = fit_one_fold( ...
        X_roi(tr,:,:),  reshape(X_roi(i,:,:), nROI, []), ...
        X_chan(tr,:,:), reshape(X_chan(i,:,:), nChan, []), ...
        y(tr), roi_feat_idx, chan_feat_idx, cfg);

    if isempty(fold); continue; end
    pred_fused(i) = fold.yhat_fused;
    pred_roi(i)   = fold.yhat_roi;
    pred_res(i)   = fold.rhat;
    best_band(i)  = fold.band_idx;
    best_roi(i)   = fold.roi_idx;
    best_lambda(i)= fold.lambda;
    nnz_chan(i)   = fold.nnz;
    beta_cell{i}  = fold.beta_full;
    chan_mask(i,:)= fold.beta_full ~= 0;

    if cfg.verbose
        fprintf('  fold %02d/%02d | band %-10s | ROI %-28s | lambda %.4g | nnz %d | y %.1f -> yhat %.1f\n', ...
            i, nSub, char(cfg.target_feats(fold.band_idx)), char(roiNames(fold.roi_idx)), ...
            fold.lambda, fold.nnz, y(i), fold.yhat_fused);
    end
end

%% ---------- Internal nested-LOSO regression metrics ----------
good = isfinite(pred_fused) & isfinite(y);
[r, perm_p, RMSE, NRMSE, MAE, R2] = regression_metrics(y(good), pred_fused(good), cfg.n_perm);

%% ---------- Re-select on the full cohort: final signature ----------
final_model = fit_final_model(X_roi, X_chan, y, roiNames, chanNames, ...
    roi_featNames, chan_featNames, roi_feat_idx, chan_feat_idx, cfg);

%% ---------- Package output ----------
bandName = strings(nSub,1); roiName = strings(nSub,1);
for i = 1:nSub
    if isfinite(best_band(i)); bandName(i) = cfg.target_feats(best_band(i)); end
    if isfinite(best_roi(i));  roiName(i)  = roiNames(best_roi(i)); end
end

OUT = struct();
OUT.subjIDs     = subjIDs;
OUT.y_true      = y;
OUT.pred_fused  = pred_fused;
OUT.pred_roi    = pred_roi;
OUT.pred_res    = pred_res;
OUT.r           = r;
OUT.perm_p      = perm_p;
OUT.RMSE        = RMSE;
OUT.NRMSE       = NRMSE;
OUT.MAE         = MAE;
OUT.R2          = R2;
OUT.roiNames    = roiNames;
OUT.chanNames   = chanNames;
OUT.per_fold    = table((1:nSub)', subjIDs(:), y(:), ...
    pred_roi(:), pred_res(:), pred_fused(:), best_band(:), bandName(:), ...
    best_roi(:), roiName(:), best_lambda(:), nnz_chan(:), ...
    'VariableNames', {'Fold','subjID','y_true', ...
    'yhat_roi','rhat_channel','yhat_fused','BandIndex','BandName', ...
    'ROIIndex','ROIName','lambda','nnz_channel'});
OUT.channel_stability = channel_stability_table(beta_cell, chan_mask, chanNames);
OUT.final_model = final_model;
OUT.cfg         = cfg;
end


%% ======================================================================
%  Local functions
%  ======================================================================

function cfg = fill_default_cfg(cfg)
% DEFAULT_CFG  Default hyper-parameters (reproduce the paper's discovery run).
d = struct();
d.target_feats = ["Rel_Delta","Rel_Theta","Rel_Alpha","Rel_Beta","Rel_Gamma"];
d.lambda_grid  = logspace(-4, 1.5, 25);   % LASSO lambda search grid
d.max_nonzero  = 10;                        % cap on selected channels
d.n_perm       = 5000;                      % permutations for the r p-value
d.rng_seed     = 222;
d.verbose      = true;
fn = fieldnames(d);
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}) || isempty(cfg.(fn{k}))
        cfg.(fn{k}) = d.(fn{k});
    end
end
end


function [roi_feat_idx, chan_feat_idx] = build_band_index(target_feats, roi_featNames, chan_featNames)
% Locate each requested band in the ROI and channel feature dimensions.
tk = normalize_string_keys(target_feats);
rk = normalize_string_keys(roi_featNames);
ck = normalize_string_keys(chan_featNames);
roi_feat_idx  = zeros(numel(target_feats),1);
chan_feat_idx = zeros(numel(target_feats),1);
for k = 1:numel(target_feats)
    hr = find(rk == tk(k), 1);
    hc = find(ck == tk(k), 1);
    if isempty(hr); error('ROI cube is missing band feature: %s', target_feats(k)); end
    if isempty(hc); error('Channel cube is missing band feature: %s', target_feats(k)); end
    roi_feat_idx(k)  = hr;
    chan_feat_idx(k) = hc;
end
end


function fold = fit_one_fold(Xroi_tr3d, xroi_te2d, Xchan_tr3d, xchan_te2d, y_tr, roi_feat_idx, chan_feat_idx, cfg)
% FIT_ONE_FOLD  One outer fold: inner-loop selection on the training subjects,
% then refit and predict the single held-out subject.  Returns [] on failure.
fold = [];
nBand = numel(cfg.target_feats);
nChan = size(Xchan_tr3d,2);
nLam  = numel(cfg.lambda_grid);

band_lambda_mse = nan(nBand, nLam);
band_roi        = nan(nBand,1);

% ----- INNER loop: choose band, ROI (Stage 1) and lambda (Stage 2) -----
for b = 1:nBand
    Xroi_b = squeeze(Xroi_tr3d(:,:,roi_feat_idx(b)));
    Xch_b  = squeeze(Xchan_tr3d(:,:,chan_feat_idx(b)));

    roiBest = stage1_select_roi(Xroi_b, y_tr);          % Stage 1 ROI-only inner LOSO
    if ~isfinite(roiBest); continue; end
    band_roi(b) = roiBest;

    band_lambda_mse(b,:) = stage2_tune_lambda( ...       % Stage 2 residual LASSO inner LOSO
        Xroi_b, Xch_b, y_tr, roiBest, cfg.lambda_grid, cfg.max_nonzero)';
end
if all(~isfinite(band_lambda_mse(:))); return; end

% best (band, lambda) by fused inner-LOSO MSE
[~, idx] = safe_nanargmin(band_lambda_mse);
[b_best, l_best] = ind2sub(size(band_lambda_mse), idx);
roi_best = band_roi(b_best);

% ----- Refit on outer-training subjects with the selected hyper-params -----
Xroi_b_tr = squeeze(Xroi_tr3d(:,:,roi_feat_idx(b_best)));
xroi_b_te = xroi_te2d(:, roi_feat_idx(b_best));

xr = Xroi_b_tr(:, roi_best);
g  = isfinite(xr) & isfinite(y_tr);
if nnz(g) < 3; return; end
b_roi = [ones(nnz(g),1), xr(g)] \ y_tr(g);             % Stage 1 univariate fit
yhat_roi_tr = nan(numel(y_tr),1);
yhat_roi_tr(g) = [ones(nnz(g),1), xr(g)] * b_roi;
resid_tr = y_tr - yhat_roi_tr;

xrt = xroi_b_te(roi_best);
if ~isfinite(xrt); return; end
yhat_roi_te = [1, xrt] * b_roi;

% channel residual model (Stage 2)
Xch_tr = squeeze(Xchan_tr3d(:,:,chan_feat_idx(b_best)));
xch_te = xchan_te2d(:, chan_feat_idx(b_best))';
[Xz_tr, Xz_te, ~] = impute_and_zscore(Xch_tr, xch_te);

gr = isfinite(resid_tr);
if nnz(gr) < 5
    fold = pack_fold(yhat_roi_te, yhat_roi_te, 0, b_best, roi_best, ...
        cfg.lambda_grid(l_best), zeros(nChan,1));
    return;
end
[B, FitInfo] = lasso(Xz_tr(gr,:), resid_tr(gr), ...
    'Lambda', cfg.lambda_grid(l_best), 'Standardize', false, ...
    'Intercept', true, 'RelTol', 1e-4, 'MaxIter', 1e5);
beta = B(:,1);
rhat_te = Xz_te * beta + FitInfo.Intercept(1);

fold = pack_fold(yhat_roi_te + rhat_te, yhat_roi_te, rhat_te, ...
    b_best, roi_best, cfg.lambda_grid(l_best), beta);
end


function fold = pack_fold(yhat_fused, yhat_roi, rhat, band_idx, roi_idx, lambda, beta_full)
fold = struct('yhat_fused',yhat_fused,'yhat_roi',yhat_roi,'rhat',rhat, ...
    'band_idx',band_idx,'roi_idx',roi_idx,'lambda',lambda, ...
    'nnz',nnz(beta_full),'beta_full',beta_full);
end


function roi_idx = stage1_select_roi(Xroi_band, y)
% STAGE1_SELECT_ROI  Pick the single ROI with the lowest leave-one-subject-out
% mean squared error of a univariate linear regression  y ~ b0 + b1*ROI.
n = numel(y);
nROI = size(Xroi_band,2);
mse = nan(nROI,1);
for r = 1:nROI
    x = Xroi_band(:,r);
    oof = nan(n,1);
    for k = 1:n
        tr = true(n,1); tr(k) = false;
        xt = x(tr); yt = y(tr);
        g = isfinite(xt) & isfinite(yt);
        if nnz(g) < 3 || ~isfinite(x(k)); continue; end
        b = [ones(nnz(g),1), xt(g)] \ yt(g);
        oof(k) = [1, x(k)] * b;
    end
    gg = isfinite(oof) & isfinite(y);
    if nnz(gg) >= max(5, ceil(0.60*n))
        mse(r) = mean((oof(gg) - y(gg)).^2);
    end
end
if all(~isfinite(mse)); roi_idx = NaN; else; [~, roi_idx] = safe_nanargmin(mse); end
end


function lambda_mse = stage2_tune_lambda(Xroi_band, Xch_band, y, roi_idx, lambda_grid, max_nonzero)
% STAGE2_TUNE_LAMBDA  With the ROI fixed, tune the residual-LASSO lambda by
% leave-one-subject-out fused-prediction MSE.  Returns MSE per lambda.
n = numel(y);
nLam = numel(lambda_grid);
oof = nan(n, nLam);
for k = 1:n
    tr = true(n,1); tr(k) = false;
    yt = y(tr);
    xr = Xroi_band(tr, roi_idx);
    xv = Xroi_band(k, roi_idx);
    g = isfinite(xr) & isfinite(yt);
    if nnz(g) < 3 || ~isfinite(xv); continue; end
    b = [ones(nnz(g),1), xr(g)] \ yt(g);
    yhat_roi_tr = nan(numel(yt),1);
    yhat_roi_tr(g) = [ones(nnz(g),1), xr(g)] * b;
    resid_tr = yt - yhat_roi_tr;
    yhat_roi_va = [1, xv] * b;

    [Xz_tr, Xz_va, ~] = impute_and_zscore(Xch_band(tr,:), Xch_band(k,:)');
    gr = isfinite(resid_tr);
    if nnz(gr) < 5; continue; end
    try
        [B, FitInfo] = lasso(Xz_tr(gr,:), resid_tr(gr), ...
            'Lambda', lambda_grid, 'Standardize', false, ...
            'Intercept', true, 'RelTol', 1e-4, 'MaxIter', 1e5);
    catch
        continue;
    end
    for l = 1:nLam
        beta = B(:,l);
        if nnz(beta) > max_nonzero; continue; end
        oof(k,l) = yhat_roi_va + (Xz_va * beta + FitInfo.Intercept(l));
    end
end
lambda_mse = nan(nLam,1);
for l = 1:nLam
    g = isfinite(oof(:,l)) & isfinite(y);
    if nnz(g) >= max(5, ceil(0.60*n))
        lambda_mse(l) = mean((oof(g,l) - y(g)).^2);
    end
end
end


function fm = fit_final_model(X_roi, X_chan, y, roiNames, chanNames, roi_featNames, chan_featNames, roi_feat_idx, chan_feat_idx, cfg)
% FIT_FINAL_MODEL  Re-run band/ROI/lambda selection on the FULL cohort and fit
% the final signature: selected band, ROI, univariate ROI regression and the
% channel-level LASSO weights.
nBand = numel(cfg.target_feats);
nChan = size(X_chan,2);
nLam  = numel(cfg.lambda_grid);

band_lambda_mse = nan(nBand, nLam);
band_roi = nan(nBand,1);
for b = 1:nBand
    Xroi_b = squeeze(X_roi(:,:,roi_feat_idx(b)));
    Xch_b  = squeeze(X_chan(:,:,chan_feat_idx(b)));
    roiBest = stage1_select_roi(Xroi_b, y);
    if ~isfinite(roiBest); continue; end
    band_roi(b) = roiBest;
    band_lambda_mse(b,:) = stage2_tune_lambda(Xroi_b, Xch_b, y, roiBest, cfg.lambda_grid, cfg.max_nonzero)';
end
if all(~isfinite(band_lambda_mse(:)))
    error('Final model selection failed: no finite fused CV-MSE on the full cohort.');
end
[~, idx] = safe_nanargmin(band_lambda_mse);
[b_best, l_best] = ind2sub(size(band_lambda_mse), idx);
roi_best = band_roi(b_best);
lambda   = cfg.lambda_grid(l_best);

% Stage 1 univariate fit on the full cohort
Xroi_b = squeeze(X_roi(:,:,roi_feat_idx(b_best)));
xr = Xroi_b(:, roi_best);
g  = isfinite(xr) & isfinite(y);
b_roi = [ones(nnz(g),1), xr(g)] \ y(g);
yhat_roi = nan(numel(y),1);
yhat_roi(g) = [ones(nnz(g),1), xr(g)] * b_roi;
resid = y - yhat_roi;

% Stage 2 residual LASSO on the full cohort (store imputation / z-score stats)
Xch_b = squeeze(X_chan(:,:,chan_feat_idx(b_best)));
mu_imp = mean(Xch_b, 1, 'omitnan'); mu_imp(~isfinite(mu_imp)) = 0;
for c = 1:nChan
    bad = ~isfinite(Xch_b(:,c));
    if any(bad); Xch_b(bad,c) = mu_imp(c); end
end
mu_z = mean(Xch_b,1);
sd_z = std(Xch_b,0,1); sd_z(sd_z==0 | ~isfinite(sd_z)) = 1;
Xz = (Xch_b - mu_z) ./ sd_z;
gr = isfinite(resid);
[Bf, FitInfof] = lasso(Xz(gr,:), resid(gr), ...
    'Lambda', lambda, 'Standardize', false, ...
    'Intercept', true, 'RelTol', 1e-4, 'MaxIter', 1e5);

fm = struct();
fm.band_name  = cfg.target_feats(b_best);
fm.lambda     = lambda;
fm.roi_name   = roiNames(roi_best);
fm.b_roi      = b_roi;            % [b0; b1] univariate ROI model
fm.mu_imp     = mu_imp;          % channel imputation means
fm.mu_z       = mu_z;            % channel z-score means
fm.sd_z       = sd_z;            % channel z-score sds
fm.beta       = Bf(:,1);         % channel LASSO weights
fm.b0_res     = FitInfof.Intercept(1);
fm.nnz        = nnz(Bf(:,1));
fm.chanNames  = chanNames;
% list the selected channels and their weights
sel = find(Bf(:,1) ~= 0);
fm.selected_channels = table(chanNames(sel), Bf(sel,1), ...
    'VariableNames', {'ChanName','Weight'});
end


function [r, p_perm, RMSE, NRMSE, MAE, R2] = regression_metrics(yt, yp, n_perm)
% REGRESSION_METRICS  Pearson r (+ permutation p), RMSE, NRMSE (vs leave-one-out
% mean predictor), MAE and R^2.
r = NaN; p_perm = NaN; RMSE = NaN; NRMSE = NaN; MAE = NaN; R2 = NaN;
n = numel(yt);
if n < 3; return; end
r = corr(yp, yt, 'Type','Pearson');
rng(333);
rp = zeros(n_perm,1);
for k = 1:n_perm
    rp(k) = corr(yp, yt(randperm(n)), 'Type','Pearson');
end
p_perm = mean(abs(rp) >= abs(r));
RMSE = sqrt(mean((yt-yp).^2));
null = nan(n,1);
for j = 1:n
    idx = true(n,1); idx(j) = false;
    null(j) = mean(yt(idx), 'omitnan');
end
NRMSE = RMSE / sqrt(mean((yt-null).^2, 'omitnan'));
MAE = mean(abs(yt-yp));
R2  = 1 - sum((yt-yp).^2)/sum((yt-mean(yt)).^2);
end


function T = channel_stability_table(beta_cell, chan_mask, chanNames)
% CHANNEL_STABILITY_TABLE  How often each channel was selected across outer
% folds, and its mean absolute weight.
nSub = numel(beta_cell);
nChan = numel(chanNames);
freq = sum(chan_mask, 1, 'omitnan')';
mab  = nan(nChan,1);
for c = 1:nChan
    tmp = nan(nSub,1);
    for i = 1:nSub
        if ~isempty(beta_cell{i}); tmp(i) = abs(beta_cell{i}(c)); end
    end
    mab(c) = mean(tmp, 'omitnan');
end
T = table((1:nChan)', chanNames(:), freq(:), mab(:), ...
    'VariableNames', {'ChanIndex','ChanName','SelectFreq','MeanAbsBeta'});
T = sortrows(T, {'SelectFreq','MeanAbsBeta'}, {'descend','descend'});
end


function [Xz_tr, Xz_te, mu_imp] = impute_and_zscore(Xtr, xte_col)
% IMPUTE_AND_ZSCORE  Mean-impute missing values and z-score channel features
% using TRAINING statistics only; apply the same transform to the test column.
mu_imp = mean(Xtr, 1, 'omitnan'); mu_imp(~isfinite(mu_imp)) = 0;
for c = 1:size(Xtr,2)
    bad = ~isfinite(Xtr(:,c));
    if any(bad); Xtr(bad,c) = mu_imp(c); end
end
xte = xte_col(:)';
for c = 1:numel(xte)
    if ~isfinite(xte(c)); xte(c) = mu_imp(c); end
end
mu = mean(Xtr,1);
sd = std(Xtr,0,1); sd(sd==0 | ~isfinite(sd)) = 1;
Xz_tr = (Xtr - mu) ./ sd;
Xz_te = (xte - mu) ./ sd;
end


function [v, idx] = safe_nanargmin(x)
flat = x(:); good = isfinite(flat);
if ~any(good); v = NaN; idx = NaN; return; end
gi = find(good);
[v, loc] = min(flat(good));
idx = gi(loc);
end
