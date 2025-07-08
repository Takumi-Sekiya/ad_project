function func_06_normalize(processed_data_dir, subject_ids, dartel_template_dir, job_params)
    
    template_file = spm_select('FPList', dartel_template_dir, '^Template_6\.nii$');
    if isempty(template_file), error('Template_6.nii not found in %s', dartel_template_dir); end

    job_input_folder = job_params.input_folder_suffix;
    job_file_pattern = job_params.file_pattern;
    job_preserve = job_params.preserve;
    job_fwhm = job_params.fwhm;

    parfor i = 1:length(subject_ids)
        spm('Defaults', 'fmri');
        spm_jobman('initcfg');
        
        sub_id = subject_ids{i};
        
        % --- 入力ファイル、流れ場、出力先を決定 ---
        input_dir = fullfile(processed_data_dir, sub_id, job_input_folder);
        flowfield_file = fullfile(processed_data_dir, sub_id, 'spm', 'seg', ['u_rc1', sub_id, '_T1w_Template.nii']);
        
        % job_params.file_pattern のワイルドカードを展開
        % 例: 'sub-*_T1w.nii' -> 'sub-001_T1w.nii'
        fname = strrep(job_file_pattern, '*', sub_id);
        image_to_normalize = fullfile(input_dir, fname);

        output_dir = fullfile(processed_data_dir, sub_id, 'spm', 'norm');
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end
        
        if ~exist(flowfield_file, 'file')
            warning('Skipping normalization for %s: Input file not found.', sub_id);
            continue;
        end

        if ~exist(image_to_normalize, 'file') || ~exist(flowfield_file, 'file')
            warning('Skipping normalization for %s: Input file or flowfield not found.', sub_id);
            continue;
        end
        
        % --- matlabbatchの作成 ---
        matlabbatch = {};
        matlabbatch{1}.spm.tools.dartel.mni_norm.template = {template_file};
        matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(1).flowfield = {flowfield_file};
        matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(1).images = {image_to_normalize};
        matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = job_preserve;
        matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = job_fwhm;
        
        % --- 実行とファイル移動 ---
        try
            % SPMは入力ファイルと同じ場所に'w'を付けて出力する
            spm_jobman('run', matlabbatch);
            
            [~, name, ext] = fileparts(image_to_normalize);
            normalized_file = ['*w', name, ext];
            source_paths = dir(fullfile(input_dir, normalized_file));
            if isempty(source_paths)
                error('No matching files found');
            end
            source_path = fullfile(source_paths.folder, source_paths.name)
            
            if exist(source_path, 'file')
                movefile(source_path, output_dir);
            end
        catch e
            warning('Normalization failed for %s (%s): %s', sub_id, fname, e.message);
        end
    end
end