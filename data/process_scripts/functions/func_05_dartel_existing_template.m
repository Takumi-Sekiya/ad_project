function func_05_dartel_existing_template(processed_data_dir, subject_ids, dartel_template_dir)
    % 既存のTemplate_1〜6.niiのリストを取得
    templates = cell(6, 1);
    for t = 1:6
        t_file = fullfile(dartel_template_dir, sprintf('Template_%d.nii', t));
        if ~exist(t_file, 'file')
            error('Template file not found: %s', t_file);
        end
        templates{t} = t_file;
    end

    % 被験者ごとに処理を実行（並列処理対応）
    parfor i = 1:length(subject_ids)
        spm('Defaults', 'fmri');
        spm_jobman('initcfg');
        
        sub_id = subject_ids{i};
        seg_dir = fullfile(processed_data_dir, sub_id, 'spm', 'seg');
        
        % 入力ファイルの確認 (rc1, rc2)
        rc1_file = fullfile(seg_dir, ['rc1', sub_id, '_T1w.nii']);
        rc2_file = fullfile(seg_dir, ['rc2', sub_id, '_T1w.nii']);
        
        % すでに流れ場が存在する場合はスキップ（必要に応じて変更）
        flowfield_file = fullfile(seg_dir, ['u_rc1', sub_id, '_T1w_Template.nii']);
        if exist(flowfield_file, 'file')
            fprintf('Flow field for %s already exists. Skipping.\n', sub_id);
            continue;
        end

        if ~exist(rc1_file, 'file') || ~exist(rc2_file, 'file')
            warning('rc files missing for %s. Skipping.', sub_id);
            continue;
        end

        fprintf('Running DARTEL (existing template) for subject: %s\n', sub_id);

        % --- matlabbatchの作成 ---
        matlabbatch = {};
        % SPMの「Run DARTEL (existing Templates)」の設定
        matlabbatch{1}.spm.tools.dartel.warp1.images = {{rc1_file}, {rc2_file}};
        matlabbatch{1}.spm.tools.dartel.warp1.settings.rform = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).rparam = [4 2 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).K = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).template = {templates{1}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).rparam = [2 1 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).K = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).template = {templates{2}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).rparam = [1 0.5 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).K = 1;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).template = {templates{3}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).rparam = [0.5 0.25 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).K = 2;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).template = {templates{4}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).rparam = [0.25 0.125 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).K = 4;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).template = {templates{5}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).rparam = [0.25 0.125 1e-06];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).K = 6;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).template = {templates{6}};
        
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.lmreg = 0.01;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.cyc = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.its = 3;

        % --- 実行 ---
        try
            spm_jobman('run', matlabbatch);
        catch e
            warning('DARTEL existing template failed for subject %s: %s', sub_id, e.message);
        end
    end
end