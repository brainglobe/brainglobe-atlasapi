# Squirrel Monkey Atlas Testing Notes

## Testing Summary

### Download URL
✅ **Working**: `https://www.nitrc.org/frs/download.php/12413/VALiDATe29.zip/?i_agree=1&download_now=1`
- File size: ~29 MB
- Version: 1.2 (latest as of Dec 2021)

### File Structure

After extraction, the archive contains:

```
VALiDATe29/
├── LICENSE.txt
├── VALiDATe-brainmask.nii.gz
├── VALiDATe-labels.txt          ← Label definitions
├── VALiDATe12-t1.nii.gz         ← Main T1 template (REFERENCE)
├── VALiDATe18-pd.nii.gz         ← Proton density template
├── VALiDATe22-t2.nii.gz         ← T2* template
├── VALiDATe3-labels.nii.gz      ← Segmentation labels (ANNOTATION)
├── VALiDATe5-exvivo-fa.nii.gz   ← Ex-vivo fractional anisotropy
├── VALiDATe6-exvivo-structural.nii.gz  ← Ex-vivo structural
├── VALiDATe7-invivo-fa.nii.gz   ← In-vivo fractional anisotropy
└── VALiDATe7-invivo-md.nii.gz   ← In-vivo mean diffusivity
```

### Label File Format

`VALiDATe-labels.txt` format:
```
<Section Header>
ID name acronym
ID name acronym
...
```

Example:
```
Gray Matter
1 l_prefrontal_cortex PFC
2 r_prefrontal_cortex PFC
...

White Matter
19 body_of_the_corpus_callosum BCC
...

Other
91 third_ventricle Ven3
...
```

### Key Findings

1. **97 total labels** (including "Not used" entries):
   - 18 Gray matter regions (9 bilateral pairs)
   - 62 White matter tracts (mostly bilateral)
   - 6 Ventricle labels
   - Some unused placeholder IDs (22, 29, 42, 92, 95)

2. **Bilateral structures**: Most regions have `l_` and `r_` prefixes
   - Sequential IDs (odd/even pairs)
   - Need hemisphere mapping

3. **Midline structures**: Some structures are bilateral (both hemispheres):
   - ID 19-21: Corpus callosum
   - ID 30: Fornix
   - ID 31-32: Commissures
   - ID 41: Optic tract

### Script Updates Needed

1. **File paths** (confirmed):
   - Reference: `VALiDATe29/VALiDATe12-t1.nii.gz`
   - Annotation: `VALiDATe29/VALiDATe3-labels.nii.gz`
   - Labels: `VALiDATe29/VALiDATe-labels.txt`

2. **Label parsing**:
   - Format: space-separated with 3 columns (ID, name, acronym)
   - Skip section headers ("Gray Matter", "White Matter", "Other")
   - Skip "Not used" entries
   - Handle lines with missing acronyms

3. **Additional references**:
   - `t2star`: `VALiDATe22-t2.nii.gz`
   - `pd`: `VALiDATe18-pd.nii.gz`
   - `invivo_fa`: `VALiDATe7-invivo-fa.nii.gz`
   - `invivo_md`: `VALiDATe7-invivo-md.nii.gz`
   - `exvivo_fa`: `VALiDATe5-exvivo-fa.nii.gz`
   - `exvivo_structural`: `VALiDATe6-exvivo-structural.nii.gz`

4. **Hemisphere handling**:
   - Create hemisphere mask (1=left, 2=right)
   - Map `l_` prefixed regions to left hemisphere
   - Map `r_` prefixed regions to right hemisphere
   - Midline structures (no prefix) map to both

## Next Steps

1. ✅ Download URL verified
2. ✅ File structure documented
3. ✅ Label format understood
4. ⏳ Update script with correct file paths
5. ⏳ Implement proper label parsing
6. ⏳ Test full atlas generation (requires Python 3.11+)
7. ⏳ Create PR

## Notes

- The current script already has flexible file detection
- Main updates needed are in the label parsing section
- The atlas appears straightforward - no major challenges like the Infant Macaque atlas
- Resolution is 300 microns as specified in the issue
- Orientation is RAS as documented in the publication
