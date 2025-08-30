%%頭部MRI画像前処理パイプライン
%
%FreeSurferによる出力ファイルを用いてSPMによるVBM, 標準化を実施する
%
%(1)パラメータ設定　:解析に必要な情報の設定を行う
%(2)処理の選択　　　:実行するステップのフラグをtrueにする
%(3)パイプライン実行:スクリプトを実行する

clear;
spm('Defaults', 'fmri');
spm_jobman('initcfg');

%%(1)パラメータ設定
%---------------------------------------------------------------
%--- 基本パス設定 ---
project_dir = '/media/sf_ad_project/data'; %プロジェクトフォルダへのパス
raw_data_dir = fullfile(project_dir, 'raw_data'); %生データ(DICOM)ディレクトリ
bids_nifti_dir = fullfile(project_dir, 'derivatives', 'bids_nifti'); %NIfTI化データ保存先
fs_subjects_dir = fullfile(project_dir, 'derivatives', 'freesurfer_subjects'); %FreeSurferのSUBJECTS_DIR
processed_data_dir = fullfile(project_dir, 'derivatives', 'processed_data');
script_dir = fullfile(project_dir, 'process_scripts');
addpath(fullfile(script_dir, 'functions'));

%--- 被験者リスト ---
prefixes = {'YGT_*', 'SND_*', 'AMC_*'}; %抽出する接頭辞のパターンを指定
d = [];

for i = 1:length(prefixes)
    d_temp = dir(fullfile(raw_data_dir, prefixes{i}));
    d = [d; d_temp];
end

isub = [d(:).isdir];
subject_ids = {d(isub).name}';
subject_ids(ismember(subject_ids,{'.','..'})) = [];

%--- ROIマスク設定 ---
%FreeSurfer(aparc+aseg.mgz)のラベル地に基づいてマスクを作成
roi_sets = {
    struct('name', 'hippocampus', 'labels', [17 53]), ...
    struct('name', 'prefrontal-cortex', 'labels', [1002 1003 1012 1014 1018 1019 1020 1026 1027 1028 1032 ...
                                                   2002 2003 2012 2014 2018 2019 2020 2026 2027 2028 2032]), ...
    struct('name', 'parietal-lobe', 'labels', [1008 1017 1022 1023 1025 1029 1031 ...
                                               2008 2017 2022 2023 2025 2029 2031]), ...
    struct('name', 'occipital-lobe', 'labels', [1005 1007 1011 1013 1021 ...
                                                2005 2007 2011 2013 2021]), ...
    struct('name', 'temporal-lobe', 'labels', [1001 1006 1007 1009 1015 1016 1030 1033 1034 ...
                                               2001 2006 2007 2009 2015 2016 2030 2033 2034]), ...
    struct('name', 'brain-stem', 'labels', 16)
    %他のROIを追加する場合はここに追記
};

%--- SPMバッチテンプレート ---
template_segment = fullfile(script_dir, 'spm_batch_templates', 'template_segment.mat');
template_dartel  = fullfile(script_dir, 'spm_batch_templates', 'template_dartel_createtemplate.mat');


%% (2) 処理の選択 (実行したいステップを true にする)
% -------------------------------------------------------------------------
flags.run_step1_dicom_to_nifti    = false; % DICOM -> NIfTI 変換
flags.run_step2_run_recon_all     = true; % FreeSurfer recon-all 実行
flags.run_step3_prepare_nifti     = true; % FreeSurfer出力 -> NIfTI/マスク作成
flags.run_step4_segment           = true; % SPM Segment
flags.run_step5_dartel_template   = true; % DARTEL Template 作成
flags.run_step6_normalize         = true; % Normalise to MNI


%% (3) パイプライン実行
% -------------------------------------------------------------------------
% --- 並列処理の準備 ---
setup_parpool();

% --- Step 1: DICOM to NIfTI --- %%% 追加 %%%
if flags.run_step1_dicom_to_nifti
    fprintf('\n===== Step 1: Converting DICOM to NIfTI files =====\n');
    func_01_dicom_to_nifti(raw_data_dir, bids_nifti_dir, subject_ids);
end

% --- Step 2: Run FreeSurfer recon-all --- %%% 追加 %%%
if flags.run_step2_run_recon_all
    fprintf('\n===== Step 2: Running FreeSurfer recon-all =====\n');
    func_02_run_recon_all(bids_nifti_dir, fs_subjects_dir, subject_ids);
end

% --- Step 3: NIfTIファイルの準備 ---
if flags.run_step3_prepare_nifti
    fprintf('\n===== Step 3: Preparing NIfTI files =====\n');
    func_03_prepare_nifti(fs_subjects_dir, processed_data_dir, subject_ids, roi_sets);
end

% --- Step 4: SPM Segment ---
if flags.run_step4_segment
    fprintf('\n===== Step 4: Running SPM Segmentation =====\n');
    func_04_segment(processed_data_dir, subject_ids, template_segment);
end

% --- Step 5: DARTEL Template 作成 ---
if flags.run_step5_dartel_template
    fprintf('\n===== Step 5: Creating DARTEL Template =====\n');
    func_05_dartel_createtemplate(processed_data_dir, subject_ids, template_dartel);
end

% --- Step 6: Normalise to MNI ---
if flags.run_step6_normalize
    fprintf('\n===== Step 6: Normalizing files to MNI space =====\n');
    dartel_template_dir = fullfile(processed_data_dir, 'dartel_template');

    % --- 正規化ジョブの定義 ---
    % ここで正規化したいファイルの種類と、そのパラメータを定義する
    % これにより、対象ごとに変調(preserve)や平滑化(fwhm)を柔軟に変更可能
    
    norm_jobs = {};
    
    % JOB 1: 元のT1w画像を正規化（平滑化あり、変調なし）
    % 可視化や、他の空間のマスクを重ねる際などに使用
    %
    norm_jobs{end+1} = struct(...
        'file_pattern', '*_T1w.nii', ...       % 対象ファイル (anatフォルダ内)
        'input_folder_suffix', 'anat', ...         % 入力ファイルがあるフォルダ
        'preserve', 1, ...                         % 1: 変調あり (体積情報を保持)
        'fwhm', [0 0 0] ...                        % 平滑化カーネルサイズ 0は平滑化なし
    );
    %

    % JOB 2: 部位マスクを正規化（平滑化なし、変調なし、最近傍補間）
    % マスク画像は値が離散的なので、補間や平滑化は行わないのが鉄則
    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-hippocampus.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         
        'fwhm', [0 0 0] ...
    );

    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-prefrontal-cortex.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         % 0: 変調なし
        'fwhm', [0 0 0] ...
    );

    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-parietal-lobe.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         
        'fwhm', [0 0 0] ...
    );

    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-occipital-lobe.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         
        'fwhm', [0 0 0] ...
    );

    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-temporal-lobe.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         
        'fwhm', [0 0 0] ...
    );

    norm_jobs{end+1} = struct(...
        'file_pattern', '*_mask-brain-stem.nii', ...
        'input_folder_suffix', 'mask', ...
        'preserve', 0, ...                         
        'fwhm', [0 0 0] ...
    );

    
    % JOB 3: 灰白質(c1)画像を正規化（平滑化あり、変調あり）
    % VBM解析の入力として最も一般的に用いられるデータ
    %
    norm_jobs{end+1} = struct(...
        'file_pattern', 'c1*_T1w.nii', ...
        'input_folder_suffix', 'spm/seg', ...
        'preserve', 0, ...
        'fwhm', [0 0 0] ...
    );
    %

    % JOB 4: 白質(c2)画像を正規化（平滑化あり、変調あり）
    %{
    norm_jobs{end+1} = struct(...
        'file_pattern', 'c2*_T1w.nii', ...
        'input_folder_suffix', 'spm/seg', ...
        'preserve', 0, ...
        'fwhm', [0 0 0] ...
    );
    %}

    % --- 上記で定義した全ジョブを実行 ---
    for i = 1:length(norm_jobs)
        job = norm_jobs{i};
        fprintf('\n--- Running Normalization Job %d/%d: %s ---\n', i, length(norm_jobs), job.file_pattern);
        func_06_normalize(processed_data_dir, subject_ids, dartel_template_dir, job);
    end
end

fprintf('\n===== Pipeline finished =====\n');

