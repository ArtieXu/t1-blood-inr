% Strict final evaluation for the two 50-frame INR candidates.
% Required environment variables:
%   T1_INR_FIXED_FILE  - fixed-CG recon_final.mat
%   T1_INR_DECAY_FILE  - CG-weight-decay recon_final.mat

clear; close all; clc;
HERE = fileparts(mfilename('fullpath'));
ROOT = fileparts(fileparts(HERE));
REF_DIR = fullfile(ROOT, 'JHU', 'T1_Blood', '20240926_ML');
addpath(fullfile(REF_DIR, 'functions'), '-begin');
assert(contains(which('Spiral_Deblur'), fullfile(REF_DIR, 'functions')), ...
    'Unexpected Spiral_Deblur: %s', which('Spiral_Deblur'));

fixed_file = getenv('T1_INR_FIXED_FILE');
decay_file = getenv('T1_INR_DECAY_FILE');
assert(isfile(fixed_file), 'Set T1_INR_FIXED_FILE to the fixed-CG recon_final.mat.');
assert(isfile(decay_file), 'Set T1_INR_DECAY_FILE to the decay-CG recon_final.mat.');

full_file = fullfile(HERE, 'results', 'full_spiral_reference.mat');
if ~isfile(full_file), build_full_spiral_reference(full_file); end
F = load(full_file, 'I_full_spiral');
H = load(fullfile(REF_DIR, 'new_initial.mat'), 'B0_unwrap', ...
    'Traj_gassp_a1s', 'h_sin_gassp_a1', 'I_deblur_unwrap_1a');
dwell = H.h_sin_gassp_a1.Par_Independent.sample_time_interval_000000;

I_fixed = load_python_volume(fixed_file);
I_decay = load_python_volume(decay_file);
I_fixed_deblur = Spiral_Deblur(I_fixed, repmat(H.B0_unwrap, [1, 1, 50]), ...
    H.Traj_gassp_a1s, dwell, 15, 1);
I_decay_deblur = Spiral_Deblur(I_decay, repmat(H.B0_unwrap, [1, 1, 50]), ...
    H.Traj_gassp_a1s, dwell, 15, 1);

loc = [106, 146];
res = 1;
r = sqrt(2) / 2;
vd_max = 15;
vd_min = 5;
TI = 50 + (0:49) * 200;
[mask_shared, ~, ~] = Vessel_Segment(F.I_full_spiral, loc, res, r, vd_max, vd_min, false);

methods = {
    'Fully sampled Spiral (B0 deblur)', F.I_full_spiral
    'MATLAB CG (B0 deblur)', H.I_deblur_unwrap_1a
    'INR 50 fixed CG (B0 deblur)', I_fixed_deblur
    'INR 50 CG decay (B0 deblur)', I_decay_deblur
};

results = struct('name', {}, 'T1_ms', {}, 'M0', {}, 'Mz', {}, 'R2', {}, ...
    'delta_T1_ms', {}, 'data_mean', {}, 'data_sd', {}, 'data_fit', {}, 'crop', {});
for idx = 1:size(methods, 1)
    I_crop = shared_crop(methods{idx, 2}, loc, res, vd_max);
    pixels = reshape(I_crop, [], 50);
    data = pixels(mask_shared(:), :);
    data_mean = double(abs(mean(data, 1)));
    data_sd = double(std(data, 1, 1));
    data_snr = data_mean ./ max(data_sd, eps);
    [T1, M0, Mz] = T1_IR_lsqr_weighted(data_mean, TI, data_snr);
    data_fit = abs(M0 + (Mz - M0) .* exp(-TI / T1));
    R2 = 1 - sum((data_mean - data_fit).^2) / ...
        sum((data_mean - mean(data_mean)).^2);
    results(end + 1) = struct('name', methods{idx, 1}, 'T1_ms', T1, ...
        'M0', M0, 'Mz', Mz, 'R2', R2, 'delta_T1_ms', NaN, ...
        'data_mean', data_mean, 'data_sd', data_sd, ...
        'data_fit', data_fit, 'crop', I_crop); %#ok<SAGROW>
end
ref_T1 = results(1).T1_ms;
for idx = 1:numel(results), results(idx).delta_T1_ms = results(idx).T1_ms - ref_T1; end

summary = table(string({results.name})', [results.T1_ms]', ...
    [results.delta_T1_ms]', [results.R2]', ...
    'VariableNames', {'method', 'T1_ms', 'delta_T1_ms_vs_full', 'R2'});
disp(summary);

out_dir = getenv('T1_INR_EVAL_OUT');
if isempty(out_dir), out_dir = fullfile(HERE, 'results', 'inr50_fullref_eval'); end
if ~isfolder(out_dir), mkdir(out_dir); end
writetable(summary, fullfile(out_dir, 'shared_mask_t1_summary.csv'));
save(fullfile(out_dir, 'shared_mask_t1_results.mat'), 'results', 'summary', ...
    'mask_shared', 'loc', 'TI', 'fixed_file', 'decay_file', 'dwell', '-v7.3');

f = figure('Color', 'w', 'Position', [100, 100, 1350, 700]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for idx = 1:numel(results)
    nexttile;
    errorbar(TI, results(idx).data_mean, results(idx).data_sd, '.', ...
        'Color', [0.1, 0.25, 0.7]); hold on;
    plot(TI, results(idx).data_fit, 'r-', 'LineWidth', 1.5);
    grid on; xlim([TI(1), TI(end)]); xlabel('TI (ms)'); ylabel('shared-ROI magnitude');
    title(sprintf('%s\nT1 %.0f ms, delta %.0f ms, R2 %.4f', ...
        results(idx).name, results(idx).T1_ms, results(idx).delta_T1_ms, results(idx).R2), ...
        'Interpreter', 'none');
end
exportgraphics(f, fullfile(out_dir, 'shared_mask_t1_curves.png'), 'Resolution', 200);
fprintf('Saved strict fully sampled evaluation to %s\n', out_dir);


function I = load_python_volume(file)
s = load(file);
assert(isfield(s, 'img_inr_physical'), '%s lacks img_inr_physical.', file);
I = squeeze(s.img_inr_physical);
if isequal(size(I), [50, 216, 216]), I = permute(I, [2, 3, 1]); end
assert(isequal(size(I), [216, 216, 50]), ...
    '%s has unsupported image size %s.', file, mat2str(size(I)));
I = double(I);
end


function I_crop = shared_crop(I, loc, res, vd_max)
radius = vd_max / res;
offset = floor(-radius / 2:radius / 2);
I_crop = I(loc(1) + offset, loc(2) + offset, :);
I_crop = imresize(I_crop, 8);
end
