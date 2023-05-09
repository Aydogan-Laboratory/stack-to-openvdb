import configparser
import os.path
from collections import namedtuple
from pathlib import Path
from typing import List

import javabridge
import numpy as np
import vtk
from fileops.cached.cached_image_file import ensure_dir
from fileops.image import OMEImageFile
from fileops.logger import get_logger
from roifile import ImagejRoi
from vtkmodules.vtkIOOpenVDB import vtkOpenVDBWriter

log = get_logger(name='export')


def bioformats_to_ndarray_zstack_timeseries(img_struct: OMEImageFile, frames: List[int], roi=None, channel=0):
    """
    Constructs a memory-intensive numpy ndarray of a whole OMEImageFile timeseries.
    Warning, it can lead to memory issues on machines with low RAM.
    """
    log.info("Exporting bioformats file to and ndarray representing a series of z-stack volumes.")

    if roi is not None:
        log.debug("Processing ROI definition that is in configuration file")
        w = abs(roi.right - roi.left)
        h = abs(roi.top - roi.bottom)
        x0 = int(roi.left)
        y0 = int(roi.top)
        x1 = int(x0 + w)
        y1 = int(y0 + h)
    else:
        log.debug("No ROI definition in configuration file")
        w = img_struct.width
        h = img_struct.height
        x0 = 0
        y0 = 0
        x1 = w
        y1 = h

    image = np.empty(shape=(len(frames), len(img_struct.zstacks), h, w), dtype=np.uint16)
    for i, frame in enumerate(frames):
        img_z = np.empty(shape=(len(img_struct.zstacks), h, w), dtype=np.uint16)
        for j, z in enumerate(img_struct.zstacks):
            log.debug(f"c={channel}, z={z}, t={frame}")
            ix = img_struct.ix_at(c=channel, z=z, t=frame)
            mdimg = img_struct.image(ix)
            img_z[j, :, :] = mdimg.image[y0:y1, x0:x1]

        # assign equalised volume into overall numpy array
        image[i, :, :, :] = img_z

    # # convert to 8-bit data and normalize intensities across whole timeseries
    image = ((image - image.min()) / (image.ptp() / 255.0)).astype(np.uint8)
    print(image.dtype)
    return image


def _ndarray_to_vtk_image(data: np.ndarray, um_per_pix=1.0, um_per_z=1.0):
    ztot, col, row = data.shape

    # For VTK to be able to use the data, it must be stored as a VTK-image.
    vtk_image = vtk.vtkImageImport()
    data_string = data.tobytes()
    vtk_image.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    vtk_image.SetDataScalarTypeToUnsignedChar()

    # dimensions of the array that data is stored in.
    vtk_image.SetNumberOfScalarComponents(1)
    vtk_image.SetScalarArrayName("density")
    vtk_image.SetDataExtent(1, row, 1, col, 1, ztot)
    vtk_image.SetWholeExtent(1, row, 1, col, 1, ztot)

    # scale data to calibration in micrometers
    vtk_image.SetDataSpacing(um_per_pix, um_per_pix, um_per_z)

    return vtk_image


def _save_vtk_image_to_disk(vtk_image, filename):
    writer = vtkOpenVDBWriter()
    writer.SetInputConnection(vtk_image.GetOutputPort())
    if os.path.exists(filename):
        os.remove(filename)
    writer.SetFileName(filename)
    writer.Update()


def save_ndarray_as_vdb(data: np.ndarray, um_per_pix=1.0, um_per_z=1.0, filename="output.vdb"):
    vtkim = _ndarray_to_vtk_image(data, um_per_pix=um_per_pix, um_per_z=um_per_z)
    _save_vtk_image_to_disk(vtkim, filename)


# ------------------------------------------------------------------------------------------------------------------
#  routines for handling of configuration files
# ------------------------------------------------------------------------------------------------------------------
ExportConfig = namedtuple('ExportConfig',
                          ['series', 'frames', 'channels', 'path', 'name', 'image_file', 'roi', 'um_per_z', ])


def _load_project_file(path) -> configparser.ConfigParser:
    prj = configparser.ConfigParser()
    prj.read(path)

    return prj


def read_config(cfg_path) -> ExportConfig:
    cfg = _load_project_file(cfg_path)

    im_series = int(cfg["DATA"]["series"])
    im_frame = cfg["DATA"]["frame"]
    im_channel = cfg["DATA"]["channel"]
    img_path = Path(cfg["DATA"]["image"])

    # process ROI path
    roi = None
    if "ROI" in cfg["DATA"]:
        roi_path = Path(cfg["DATA"]["ROI"])
        if not roi_path.is_absolute():
            roi_path = cfg_path.parent / roi_path
            roi = ImagejRoi.fromfile(roi_path)
            im_frame = roi.t_position

    img_file = OMEImageFile(img_path.as_posix(), image_series=im_series)

    return ExportConfig(series=im_series,
                        frames=range(img_file.n_frames) if im_frame == "all" else [int(im_frame)],
                        channels=range(img_file.n_channels) if im_channel == "all" else [int(im_channel)],
                        path=cfg_path.parent,
                        name=cfg_path.name,
                        image_file=img_file,
                        um_per_z=float(cfg["DATA"]["um_per_z"]) if "um_per_z" in cfg["DATA"] else img_file.um_per_z,
                        roi=roi)


if __name__ == "__main__":
    base_path = Path(".")
    cfg_path_list = [
        base_path / "example.cfg",
    ]
    for cfg_path in cfg_path_list:
        log.info(f"Reading configuration file {cfg_path}")
        cfg = read_config(cfg_path)

        for ch in cfg.channels:
            # prepare path for exporting data
            export_path = ensure_dir(cfg_path.parent / "openvdb" / f"ch{ch:01d}")

            frames = list(range(cfg.image_file.n_frames))
            vol_timeseries = bioformats_to_ndarray_zstack_timeseries(cfg.image_file, frames, roi=cfg.roi, channel=ch)

            for fr, vol in enumerate(vol_timeseries):
                if fr not in cfg.frames:
                    continue
                vtkim = _ndarray_to_vtk_image(vol, um_per_pix=cfg.image_file.um_per_pix, um_per_z=cfg.um_per_z)
                _save_vtk_image_to_disk(vtkim, export_path / f"vol_ch{ch:01d}_fr{fr:03d}.vdb")

    javabridge.kill_vm()
