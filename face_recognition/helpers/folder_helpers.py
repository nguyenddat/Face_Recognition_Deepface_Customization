import os

from ..helpers.logger import logger

def initialize_folder() -> None:
    home = get_home()
    weights_path = os.path.join(home, "weights")
    
    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
        logger.info(f"Directory {weights_path} has been created")

def get_home():
    return str(os.path.join(os.getcwd()))