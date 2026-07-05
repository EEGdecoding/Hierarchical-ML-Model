function [subjIDs, subjKey] = normalize_subject_ids(subjIDs_in, nSub, prefix)
% NORMALIZE_SUBJECT_IDS  Return string subject IDs and their canonical keys.
% If the stored IDs are missing or do not match the number of subjects,
% synthetic IDs (prefix + zero-padded index) are generated instead.
if nargin < 3; prefix = "Sub"; end
if isempty(subjIDs_in)
    subjIDs = string(compose("%s%03d", prefix, (1:nSub)'));
else
    subjIDs = string(subjIDs_in(:));
    if numel(subjIDs) ~= nSub
        subjIDs = string(compose("%s%03d", prefix, (1:nSub)'));
    end
end
subjKey = normalize_string_keys(subjIDs);
end
