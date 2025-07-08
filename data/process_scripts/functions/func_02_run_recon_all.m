function func_02_run_recon_all(nifti_dir, fs_subjects_dir, subject_ids)
    fs_home = '/usr/local/freesurfer';
    setup_cmd = sprintf('export FREESURFER_HOME=%s && source $FREESURFER_HOME/SetUpFreeSurfer.sh', fs_home);

    parfor i = 1:length(subject_ids)
        sub_id = subject_ids{i};
        fprintf('Step 2: Starting recon-all for subject: %s\n', sub_id);

        input_nifti = fullfile(nifti_dir, sub_id, [sub_id, '_T1w.nii']);
        output_fs_dir = fullfile(fs_subjects_dir, sub_id);

        if exist(fullfile(output_fs_dir, 'scripts', 'recon-all.done'), 'file')
            fprintf('recon-all for %s has already completed. Skipping.\n', sub_id);
            continue;
        end

        if ~exist(input_nifti, 'file')
            fprintf('recon-all for %s has already completed. Skipping.\n', sub_id);
            continue;
        end

        recon_cmd = sprintf('recon-all -s %s -i %s -all', sub_id, input_nifti);

        log_dir = fullfile(fs_subjects_dir, 'logs');
        if ~exist(log_dir, 'dir'), mkdir(log_dir); end
        log_file = fullfile(log_dir, [sub_id, 'recon-all.log']);

        full_cmd = sprintf('bash -c "%s && export SUBJECTS_DIR=%s && %s > %s 2>&1"', ...
                            setup_cmd, fs_subjects_dir, recon_cmd, log_file);
        
        status = system(full_cmd);

        if status == 0
            fprintf('recon-all for %s finished successfully.\n', sub_id);
        else
            warning('recon-all for %s may have failed. Check log file: %s', sub_id, log_file);
        end
    end
end