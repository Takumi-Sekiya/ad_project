function func_01_dicom_to_nifti(raw_data_dir, nifti_out_dir, subject_ids)
    parfor i = 1:length(subject_ids)
        spm('Defaults', 'fmri');
        spm_jobman('initcfg');

        sub_id = subject_ids{i};
        fprintf('Step 1: Converting DICOM to NIFTI for subject: %s\n', sub_id);

        dicom_in_path = fullfile(raw_data_dir, sub_id);
        nifti_out_path = fullfile(nifti_out_dir, sub_id);

        if ~exist(nifti_out_path, 'dir')
            mkdir(nifti_out_path);
        end

        output_nifti = fullfile(nifti_out_path, [sub_id, '_T1w.nii']);

        if exist(output_nifti, 'file')
            fprintf('NIFTI file for %s already exists. Skipping.\n', sub_id);
            continue;
        end

        dicom_files = spm_select('FPList', dicom_in_path, '.*');
        if isempty(dicom_files)
            warning('No DICOM files found for %s. Skipping.', sub_id);
            continue;
        end

        converted_file = spm_select('FPList', nifti_out_path, '^f.*\.nii$');
        if isempty(converted_file)
            converted_file = spm_select('FPList', nifti_out_path, '.*\.nii$');
        end

        if ~isempty(converted_file)
            movefile(strtrim(converted_file(1,:)), output_nifti);
        else
            warning('NIfTI conversion failed for %s.', sub_id);
        end
    end
end