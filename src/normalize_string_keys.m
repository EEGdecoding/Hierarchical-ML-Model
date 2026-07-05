function out = normalize_string_keys(x)
% NORMALIZE_STRING_KEYS  Canonical form for matching names/IDs across files:
% lower-case, trimmed, with whitespace/underscores/hyphens and other
% non-word characters removed.  e.g. "Rel_Alpha" and "rel alpha" both map to
% "relalpha".
x = string(x(:));
x = lower(strtrim(x));
x = regexprep(x, '[\s\-_]+', '');
x = regexprep(x, '[^\w]', '');
out = x;
end
