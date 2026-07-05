function [X_chan, subjIDs, chanNames, featNames, y] = load_channel_cube(mat_path)
% LOAD_CHANNEL_CUBE  Load a channel feature cube saved as a struct 'channel_data'.
%
%   channel_data fields:
%     X_chan     [nSub x nChan x nFeat]  channel-level features
%     subjIDs    [nSub x 1]              subject identifiers
%     chanNames  [nChan x 1]             channel names
%     featNames  [nFeat x 1]             feature/band names (e.g. Rel_Alpha)
%     pctHAMD    [nSub x 1]              outcome (percent HAMD reduction)
S = load(mat_path);
if isfield(S,'channel_data')
    channel_data = S.channel_data;
else
    fns = fieldnames(S);
    if numel(fns)==1 && isstruct(S.(fns{1}))
        channel_data = S.(fns{1});
    else
        error('Cannot recognise channel mat structure: %s', mat_path);
    end
end
X_chan    = channel_data.X_chan;
subjIDs   = string(channel_data.subjIDs(:));
chanNames = string(channel_data.chanNames(:));
featNames = string(channel_data.featNames(:));
y         = channel_data.pctHAMD(:);
end
