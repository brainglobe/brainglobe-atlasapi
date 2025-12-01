# SWC Female Rat – Pre-packaging pipeline

This folder contains scripts and instructions to prepare the SWC female rat atlas
inputs used by `atlas_scripts/swc_female_rat.py`.

The overall workflow is:

1. **Prepare WHS template + annotation in ASR space**
   Run `prepare_template_and_annotation.py` to generate:
   - `WHS_SD_T2star_clean_ASR.nii.gz`
   - `WHS_SD_annotation_ASR.nii.gz`

2. **Pre-align SWC template in napari (template-builder plugin)**
   This step is interactive and *not* scripted here.

3. **Run ANTs registration (WHS → SWC template)**
   Use `run_ants_registration.sh`.

4. **Apply transforms to the WHS annotation**
   Use `apply_ants_transforms.sh`.

5. **Clean the annotation in SWC template space**
   Use `clean_annotation.py`.

6. **Run final atlas packaging**
   Use `atlas_scripts/swc_female_rat.py` as in the main repository.

## 1. Prepare WHS template + annotation

From the repository root (or this folder, adjusting paths), run:

```bash
python atlas_scripts/pre_packaging/swc_female_rat/prepare_template_and_annotation.py
```

This expects the original WHS NIfTI volumes to be present in the working
directory, and produces:

- `WHS_SD_T2star_clean_ASR.nii.gz`
- `WHS_SD_annotation_ASR.nii.gz`

## 2. Pre-align SWC template in napari (template-builder)

This step is interactive and follows the BrainGlobe template-builder
documentation:

- Tutorial: `https://brainglobe.info/tutorials/template-builder-pre-align.html`

### 2.1. Launch napari and load SWC template

In Napari iPython:

```python
import nibabel as nib
from napari import Viewer

tmpl_img = nib.load("template_sharpen_shapeupdate.nii.gz")

viewer = Viewer()
viewer.add_image(tmpl_img, name="swc_template")
```

Or via the napari GUI by opening `template_sharpen_shapeupdate.nii.gz` directly.

### 2.2. Using the template-builder plugin

Within napari:

- Reorient the template as needed.
- Add a mask (you may need to threshold; a starting value around 0.4 often works).
- Add corresponding landmark points.
- Align to your chosen reference.
- Save results as:
  - `template_sharpen_shapeupdate_orig-asr_aligned.tif`
  - `template_sharpen_shapeupdate_orig-asr_aligned.nii.gz` (required for later steps)

Make sure the aligned NIfTI is in the same directory where you will run the
registration scripts.

## 3. ANTs registration

Registration is performed with `antsRegistration` using:

- **Fixed**: `template_sharpen_shapeupdate_orig-asr_aligned.nii.gz`
- **Moving**: `WHS_SD_T2star_clean_ASR.nii.gz`

From this folder:

```bash
chmod +x run_ants_registration.sh
./run_ants_registration.sh
```

This produces:

- `W2T_0GenericAffine.mat`
- `W2T_1Warp.nii.gz`
- `W2T_warped.nii.gz`

## 4. Apply transforms to WHS annotation

Use `antsApplyTransforms` to bring the WHS annotation into SWC template space:

```bash
chmod +x apply_ants_transforms.sh
./apply_ants_transforms.sh
```

This expects:

- `WHS_SD_annotation_ASR.nii.gz`
- `template_sharpen_shapeupdate_orig-asr_aligned.nii.gz`
- `W2T_0GenericAffine.mat`
- `W2T_1Warp.nii.gz`

and produces:

- `WHS_SD_annotation_template_space.nii.gz`

## 5. Clean the annotation

Run:

```bash
python atlas_scripts/pre_packaging/swc_female_rat/clean_annotation.py
```

This:

- Builds a mask from `template_sharpen_shapeupdate_orig-asr_aligned.nii.gz`.
- Keeps only the largest connected component of the mask.
- Applies that mask to `WHS_SD_annotation_template_space.nii.gz`.
- Removes a set of manually chosen region IDs.
- Saves:
  - `WHS_SD_annotation_template_space_cleaned.nii.gz`
  - `clean_mask.nii.gz` (for inspection/debugging)

If necessary, you can adjust:

- The intensity threshold used to build the mask.
- The list of region IDs to remove.

Both are defined near the top of `clean_annotation.py`.

## 6. Atlas packaging

Once the cleaned annotation and corresponding template are ready, run the main
packaging script (see its own documentation and CLI options):

```bash
python atlas_scripts/swc_female_rat.py
```
