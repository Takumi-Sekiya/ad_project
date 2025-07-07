function func_01_prepare_nifti(fs_subjects_dir, output_root_dir, subject_ids, roi_sets)
    parfor i = 1:length(subject_ids)
        sub_id = subject_ids{i};
        fprintf('Processing subject: %s\n', sub_id);
        
        fs_sub_dir = fullfile(fs_subjects_dir, sub_id, 'mri');
        
        % --- 出力ディレクトリ作成 ---
        anat_dir = fullfile(output_root_dir, sub_id, 'anat');
        mask_dir = fullfile(output_root_dir, sub_id, 'mask');
        if ~exist(anat_dir, 'dir'), mkdir(anat_dir); end
        if ~exist(mask_dir, 'dir'), mkdir(mask_dir); end
        
        % --- 1. T1w画像 (orig.mgz) を NIfTI に変換 ---
        t1w_mgz = fullfile(fs_sub_dir, 'orig.mgz');
        t1w_nii = fullfile(anat_dir, [sub_id, '_T1w.nii']);
        if exist(t1w_mgz, 'file') && ~exist(t1w_nii, 'file')
            cmd = sprintf('mri_convert %s %s', t1w_mgz, t1w_nii);
            system(cmd);
        end
        
        % --- 2. ROIマスク (aparc+aseg.mgz) を作成 ---
        aseg_mgz = fullfile(fs_sub_dir, 'aparc+aseg.mgz');
        if exist(aseg_mgz, 'file')
            for r = 1:length(roi_sets)
                roi = roi_sets{r};
                mask_nii = fullfile(mask_dir, [sub_id, '_mask-', roi.name, '.nii']);
                
                if ~exist(mask_nii, 'file')
                    match_str = sprintf('%d ', roi.labels);
                    cmd = sprintf('mri_binarize --i %s --match %s --o %s', aseg_mgz, match_str, mask_nii);
                    system(cmd);
                end
            end
        end
    end
end