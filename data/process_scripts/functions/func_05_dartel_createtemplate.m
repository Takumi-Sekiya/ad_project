function func_05_dartel_createtemplate(processed_data_dir, subject_ids, template_path)
    s = load(template_path);
    matlabbatch = s.matlabbatch;

    rc1_files = cell(length(subject_ids), 1);
    rc2_files = cell(length(subject_ids), 1);
    count = 0;
    for i = 1:length(subject_ids)
        sub_id = subject_ids{i};
        seg_dir = fullfile(processed_data_dir, sub_id, 'spm', 'seg');
        rc1_file = fullfile(seg_dir, ['rc1', sub_id, '_T1w.nii']);
        rc2_file = fullfile(seg_dir, ['rc2', sub_id, '_T1w.nii']);
        
        if exist(rc1_file, 'file') && exist(rc2_file, 'file')
            count = count + 1;
            rc1_files{count, 1} = rc1_file;
            rc2_files{count, 1} = rc2_file;
        end
    end
    rc1_files = rc1_files(1:count);
    rc2_files = rc2_files(1:count);
    
    if isempty(rc1_files)
        error('No rc1/rc2 files found. Run segmentation first.');
    end

    matlabbatch{1}.spm.tools.dartel.warp.images = {rc1_files, rc2_files};
    
    % --- 出力先ディレクトリ ---
    dartel_dir = fullfile(processed_data_dir, 'dartel_template');
    if ~exist(dartel_dir, 'dir'), mkdir(dartel_dir); end
    
    fprintf('Running DARTEL: Create Template with %d subjects.\n', length(rc1_files));
    spm_jobman('run', matlabbatch);
    
    % --- ファイルを移動 ---
    % DARTELは入力ファイル(rc1/rc2)と同じ場所にu_*とTemplate_*を出力する
    [source_dir, ~, ~] = fileparts(rc1_files{1});
    movefile(fullfile(source_dir, 'u_*.nii'), dartel_dir);
    movefile(fullfile(source_dir, 'Template_*.nii'), dartel_dir);
    
    fprintf('DARTEL files moved to %s\n', dartel_dir);
end