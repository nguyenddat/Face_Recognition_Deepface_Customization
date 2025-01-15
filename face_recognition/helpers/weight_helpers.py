import os
from typing import *
import zipfile
import bz2

import gdown

from helpers import folder_helpers, package_helpers
from helpers.logger import logger

tf_version = package_helpers.get_tf_major_version()
if tf_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]

def download_weights_if_necessary(
    file_name: str,
    source_url: str,
    compress_type: Optional[str] = None
) -> str:
    home = folder_helpers.get_home()
    
    target_file = os.path.normpath(os.path.join(home, "weights", file_name))
    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    if compress_type is not None and compress_type not in ALLOWED_COMPRESS_TYPES:
        raise ValueError(f"unimplemented compress type - {compress_type}")
    
    try:
        logger.info(f"🔗 {file_name} will be downloaded from {source_url} to {target_file}...")

        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)

    except Exception as err:
        raise ValueError(
            f"⛓️‍💥 An exception occurred while downloading {file_name} from {source_url}. "
            f"Consider downloading it manually to {target_file}."
        ) from err
    
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
            logger.info(f"{target_file}.zip unzipped")
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        logger.info(f"{target_file}.bz2 unzipped")

    return target_file

def load_model_weights(model: Sequential, weight_file: str) -> Sequential:
    """
    args:
        model Sequential: pre-built model
        weight_file : path of weights
    returns:
        model Sequential: pre-built model with updated weights
    """
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
        ) from err
    return model