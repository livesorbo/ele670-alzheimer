import os
import nibabel as nib

# Rotmappe der alle OASIS disc-mappene ligger
data_dir = "data"

converted = 0
skipped = 0

for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith("_t88_masked_gfc.hdr"):
            hdr_path = os.path.join(root, f)
            img_path = hdr_path.replace(".hdr", ".img")

            if not os.path.exists(img_path):
                print(f" Mangler .img for {hdr_path}, hopper over.")
                continue

            # lagres i /data
            subj_id = f.replace(".hdr", "")
            out_path = os.path.join(data_dir, subj_id + ".nii.gz")

            if os.path.exists(out_path):
                print(f" Hopper over (finnes allerede): {out_path}")
                skipped += 1
                continue

            # Konverter
            try:
                img = nib.load(hdr_path)
                nib.save(img, out_path)
                print(f" Konvertert: {out_path}")
                converted += 1
            except Exception as e:
                print(f" Feil under konvertering av {hdr_path}: {e}")

print(f"\n  Konvertert {converted} filer, hoppet over {skipped} eksisterende.")
