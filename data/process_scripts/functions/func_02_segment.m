function func_02_segment(processed_data_dir, subject_ids, template_path)
    s = load(template_path);
    matlabbatch_template = s.matlabbatch;

    parfor i = 1:length(subject_ids)
        spm('Defaults', 'fmri');
        spm_jobman('initcfg');
        
        sub_id = subject_ids{i};
        fprintf('Segmenting subject: %s\n', sub_id);
        
        t1w_nii = fullfile(processed_data_dir, sub_id, 'anat', [sub_id, '_T1w.nii']);
        
        if ~exist(t1w_nii, 'file'), continue; end
        
        % --- matlabbatchの準備 ---
        matlabbatch = matlabbatch_template;
        matlabbatch{1}.spm.spatial.preproc.channel.vols = {[t1w_nii, ',1']};
        
        % --- 出力ディレクトリの指定と作成 ---
        % SPMは出力先を指定できないため、実行後にファイルを移動する
        output_dir = fullfile(processed_data_dir, sub_id, 'spm', 'seg');
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end
        
        % --- 実行 ---
        try
            spm_jobman('run', matlabbatch);
            
            % --- ファイルの移動 ---
            [anat_path, ~, ~] = fileparts(t1w_nii);
            prefixes = {'BiasField_', 'c1', 'c2', 'c3', 'c4', 'c5', 'm', 'rc1', 'rc2'};
            base_name = [sub_id, '_T1w.nii'];
            seg_mat_name = [sub_id, '_T1w_seg8.mat'];

            for p = 1:length(prefixes)
                src = fullfile(anat_path, [prefixes{p}, base_name]);
                if exist(src, 'file'), movefile(src, output_dir); end
            end
            src_mat = fullfile(anat_path, seg_mat_name);
            if exist(src_mat, 'file'), movefile(src_mat, output_dir); end
            
        catch e
            warning('Segmentation failed for subject %s: %s', sub_id, e.message);
        end
    end
end