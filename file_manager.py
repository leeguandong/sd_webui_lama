import os
from modules import shared
from datetime import datetime
from pathlib import Path


def get_webui_setting(key, default):
    value = shared.opts.data.get(key, default)

    if not isinstance(value, type(default)):
        value = default
    return value


class FileManager:
    def __init__(self) -> None:
        self.update_ia_outputs_dir()

    def update_ia_outputs_dir(self) -> None:
        """Update lama outputs directory.

        Returns:
            None
        """
        config_save_folder = get_webui_setting("lama_save_folder", "lama")
        self.folder_is_webui = False if config_save_folder == "lama" else True
        if config_save_folder == "lama":
            self._outputs_dir = os.path.join(shared.data_path, "outputs", config_save_folder,
                                             datetime.now().strftime("%Y-%m-%d"))
        else:
            try:
                webui_save_folder = Path(
                    get_webui_setting("outdir_img2img_samples", os.path.join("outputs", "img2img-images")))
                if webui_save_folder.is_absolute():
                    self._outputs_dir = os.path.join(str(webui_save_folder), datetime.now().strftime("%Y-%m-%d"))
                else:
                    self._outputs_dir = os.path.join(shared.data_path, str(webui_save_folder),
                                                     datetime.now().strftime("%Y-%m-%d"))
            except Exception:
                self._outputs_dir = os.path.join(shared.data_path, "outputs", "img2img-images",
                                                 datetime.now().strftime("%Y-%m-%d"))

    @property
    def outputs_dir(self) -> str:
        """Get lama outputs directory.

        Returns:
            str: lama outputs directory
        """
        self.update_ia_outputs_dir()
        if not os.path.isdir(self._outputs_dir):
            os.makedirs(self._outputs_dir, exist_ok=True)
        return self._outputs_dir

    @property
    def savename_prefix(self) -> str:
        """Get lama savename prefix.

        Returns:
            str: lama savename prefix
        """
        config_save_folder = get_webui_setting("lama_save_folder", "lama")
        self.folder_is_webui = False if config_save_folder == "lama" else True
        basename = "lama-" if self.folder_is_webui else ""

        return basename + datetime.now().strftime("%Y%m%d-%H%M%S")


file_manager = FileManager()
