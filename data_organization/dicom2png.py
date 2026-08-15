import pydicom
import numpy as np
from PIL import Image
import os
import argparse

def dicom_to_png(dicom_path, png_path):
    # Load DICOM
    ds = pydicom.dcmread(dicom_path)

    # Extract pixel data
    pixel_array = ds.pixel_array

    # Handle modality LUT or VOI LUT if present
    try:
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(pixel_array, ds)
    except Exception:
        pass

    try:
        pixel_array = pydicom.pixel_data_handlers.util.apply_voi_lut(pixel_array, ds)
    except Exception:
        pass

    pixel_array = pixel_array.astype(float)

    if 'PhotometricInterpretation' in ds:
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array

    pixel_array -= pixel_array.min()
    pixel_array /= pixel_array.max()
    pixel_array *= 255
    pixel_array = pixel_array.astype(np.uint8)
        
    img = Image.fromarray(pixel_array)
    # Convert grayscale or multi-channel
    if len(pixel_array.shape) == 2:
        img = Image.fromarray(pixel_array)
    else:
        # If image has multiple channels
        img = Image.fromarray(pixel_array[..., 0:3])

    img.save(png_path)
    print(f"Saved PNG to {png_path}")

def is_negative(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    try:
        if(ds[0x2050,0x0020]).value == 'INVERSE':
            if 'PhotometricInterpretation' in ds:
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    print("Photometric")
                else:
                    exit()
            return True
    except KeyError:
        pass       
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversion from DICOM to png")
    parser.add_argument("--dicomsdir", type=str, required=True, help="DICOMs directory")
    parser.add_argument('--outputdir', type=str, required=True, help='Output directory')

    args = parser.parse_args()
    dicomsDir = args.dicomsdir
    outputDir = args.outputdir

    nFiles = 0
    for (root, _, files) in os.walk(dicomsDir):
        for f in files:
            if f.endswith(".dcm"):
                dcmFile = os.path.join(root,f)
                outputFile = f.replace(".dcm", ".png")
                outputFile = os.path.join(outputDir, outputFile)
                if not os.path.exists(outputFile):
                    dicom_to_png(dcmFile, outputFile)
                nFiles += 1
