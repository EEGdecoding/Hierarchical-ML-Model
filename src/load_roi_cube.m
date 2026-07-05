function [X_roi, subjIDs, roiNames, featNames, y] = load_roi_cube(mat_path)
% LOAD_ROI_CUBE  Load an ROI feature cube saved as a struct 'roi_data'.
%
%   roi_data fields:
%     X_roi      [nSub x nROI x nFeat]  ROI-level features
%     subjIDs    [nSub x 1]             subject identifiers
%     roiNames   [nROI x 1]             ROI names
%     featNames  [nFeat x 1]            feature/band names (e.g. Rel_Alpha)
%     pctHAMD    [nSub x 1]             outcome (percent HAMD reduction)
S = load(mat_path);
if isfield(S,'roi_data')
    roi_data = S.roi_data;
else
    fns = fieldnames(S);
    if numel(fns)==1 && isstruct(S.(fns{1}))
        roi_data = S.(fns{1});
    else
        error('Cannot recognise ROI mat structure: %s', mat_path);
    end
end
X_roi     = roi_data.X_roi;
subjIDs   = string(roi_data.subjIDs(:));
roiNames  = string(roi_data.roiNames(:));
featNames = string(roi_data.featNames(:));
y         = roi_data.pctHAMD(:);
end
