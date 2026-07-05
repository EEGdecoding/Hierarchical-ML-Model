function make_signature_scatter(y_pred, y_true, save_path, r_show, p_show)
% MAKE_SIGNATURE_SCATTER  Predicted vs. true pctHAMD scatter with a regression
% line, 95% CI band and a y=x reference. Saves a 600-dpi PNG (and a vector PDF
% when possible).
%
%   y_pred, y_true  predicted and observed pctHAMD (%)
%   save_path       output PNG path
%   r_show, p_show  Pearson r and permutation p to annotate (optional)

if nargin < 4; r_show = NaN; end
if nargin < 5; p_show = NaN; end

good = isfinite(y_pred) & isfinite(y_true);
yp = y_pred(good); yp = yp(:);
yt = y_true(good); yt = yt(:);
if numel(yp) < 3
    warning('Not enough points to plot.'); return;
end

point_color = [78 120 169]/255;
line_color  = [55 100 160]/255;
ci_color    = [202 218 232]/255;

axis_min = floor(min([yp; yt])) - 5;
axis_max = ceil(max([yp; yt])) + 5;
lims = [axis_min axis_max];

% regression line + CI
X = [ones(size(yp)), yp];
b = X \ yt;
resid = yt - X*b;
df = max(1, numel(yp)-2);
s_err = sqrt(sum(resid.^2)/df);
tv = tinv(0.975, df);
denom = sum((yp - mean(yp)).^2);
xl = linspace(lims(1), lims(2), 200)';
yl = [ones(size(xl)), xl]*b;
if denom > 0
    ci = tv*s_err*sqrt(1/numel(yp) + (xl-mean(yp)).^2/denom);
else
    ci = zeros(size(xl));
end

figure('Color','w','Units','centimeters','Position',[2 2 9 9],'Renderer','painters');
hold on;
fill([xl; flipud(xl)], [yl+ci; flipud(yl-ci)], ci_color, 'EdgeColor','none');
plot(xl, yl, '-', 'Color', line_color, 'LineWidth', 1.8);
plot(lims, lims, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
scatter(yp, yt, 55, 'MarkerFaceColor', point_color, 'MarkerEdgeColor','w', 'LineWidth',0.5);
xlim(lims); ylim(lims); axis square; box off;

xlabel('Predicted pctHAMD (%)','FontName','Arial','FontSize',9);
ylabel('True pctHAMD (%)','FontName','Arial','FontSize',9);
set(gca,'FontName','Arial','FontSize',8,'LineWidth',0.8,'TickDir','out','Layer','top');

if p_show < 0.001
    p_str = 'perm \itp\rm < 0.001';
elseif isnan(p_show)
    p_str = '';
else
    p_str = sprintf('perm \\itp\\rm = %.3f', p_show);
end
txt = {sprintf('\\itn\\rm = %d', numel(yp)), sprintf('\\itr\\rm = %.3f', r_show)};
if ~isempty(p_str); txt{end+1} = p_str; end
text(lims(1)+0.05*diff(lims), lims(2)-0.05*diff(lims), txt, ...
    'HorizontalAlignment','left','VerticalAlignment','top', ...
    'Interpreter','tex','FontName','Arial','FontSize',9,'BackgroundColor','w','Margin',1);

[folder, base, ~] = fileparts(save_path);
if ~isempty(folder) && ~exist(folder,'dir'); mkdir(folder); end
try
    exportgraphics(gcf, save_path, 'Resolution', 600, 'BackgroundColor','white');
    exportgraphics(gcf, fullfile(folder, [base '.pdf']), 'ContentType','vector', 'BackgroundColor','white');
catch
    print(gcf, save_path, '-dpng', '-r600');
end
close(gcf);
end
