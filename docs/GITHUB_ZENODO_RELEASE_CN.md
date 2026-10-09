# GitHub + Zenodo 发布操作（v1.0.0）

## A. GitHub 新仓库

建议仓库名：

`definition-dependent-low-muscle-prediction`

建议描述：

`Reproducibility resources for definition-dependent low-muscle prediction with U.S. temporal validation and frozen KNHANES external validation.`

步骤：
1. 在 GitHub 新建 **Public** repository，先不要勾选自动生成 README/LICENSE。
2. 解压 `Paper3_GitHub_Repository_Ready_v1.0.0.zip`。
3. 将解压后的**文件夹内部内容**上传到仓库根目录。
4. 上传前再次确认：
   - `data_private/` 只有 `README.md`；
   - 没有 `.sav/.rds/.xpt`；
   - 没有 `knhanes_*_paper3_ready.csv.gz`；
   - 没有任何作者个人电脑的绝对本地路径。
5. Commit message建议：`Public reproducibility release v1.0.0`
6. 检查 README、CITATION.cff、LICENSE、`.zenodo.json` 是否正常显示。

## B. Zenodo

推荐使用 GitHub-Zenodo integration：
1. Zenodo 登录后连接 GitHub。
2. Zenodo → GitHub → Sync，找到新仓库并启用。
3. 回到 GitHub 创建 Release：
   - Tag: `v1.0.0`
   - Release title: `v1.0.0 — US temporal and KNHANES external validation`
   - Release notes 可直接使用 `RELEASE_NOTES.md`
4. Zenodo会为启用的 GitHub repository 的新 release 创建归档记录。
5. 检查作者、ORCID、标题、license、keywords、description 后发布/确认记录。
6. 记录 **version-specific DOI**，把它写入最终投稿手稿的 Code availability。
7. DOI生成后，可在 README 中增加 DOI badge，但不是投稿前的必要条件。

## C. 稿件中的最终 Code availability

DOI出来后建议写：

`Frozen model objects, analysis code, model specifications, scoring functions, development-defined thresholds, aggregate U.S. temporal-validation results, and KNHANES external-validation code and non-participant-level outputs are publicly available in the version-controlled GitHub repository (https://github.com/xukaifs/definition-dependent-low-muscle-prediction) and are permanently archived in Zenodo (version-specific DOI: [NEW DOI]). Participant-level KNHANES analytic data are not redistributed in accordance with Korea Disease Control and Prevention Agency data-use requirements.`

## D. 以后如果修改文件

不要覆盖已发布的 Zenodo v1.0.0 文件。
如需实质性修改，发布 GitHub `v1.0.1` / `v1.1.0`，让 Zenodo生成新的 version DOI。
