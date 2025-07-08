function func_01_dicom_to_nifti(raw_data_dir, nifti_out_dir, subject_ids)
    parfor i = 1:length(subject_ids)
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

        if ~exist(dicom_in_path, 'dir')
            warning('Input DICOM directory not found for %s. Skipping.', sub_id);
            continue;
        end

        cmd = sprintf('dcm2niix -o %s -f %%i_T1w -b y -z n %s', ...
                      nifti_out_path, ...
                      dicom_in_path);
        
        [status, cmdout] = system(cmd);

        if status == 0
            fprintf('dcm2niix completed for subject %s.\n', sub_id);
        else
            warning('dcm2niix may have failed for subject %s. See output below:\n%s', sub_id, cmdout);
        end
    end
end