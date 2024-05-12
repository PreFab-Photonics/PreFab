"""Provides the Device class for representing photonic devices."""

import base64
import io
import json
import os
from typing import Optional

import cv2
import gdstk
import matplotlib.pyplot as plt
import numpy as np
import requests
import toml
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import BaseModel, Field, conint, root_validator, validator
from tqdm import tqdm

from . import geometry
from .models import Model


class BufferSpec(BaseModel):
    """
    Defines the specifications for a buffer zone around a device.

    This class is used to specify the mode and thickness of a buffer zone that is added
    around the device geometry. The buffer zone can be used for various purposes such as
    providing extra space for device fabrication processes or for ensuring that the
    device is isolated from surrounding structures.

    Parameters
    ----------
    mode : dict[str, str]
        A dictionary that defines the buffer mode for each side of the device
        ('top', 'bottom', 'left', 'right'), where 'constant' is used for isolated
        structures and 'edge' is utilized for preserving the edge, such as for waveguide
        connections.
    thickness : conint(gt=0)
        The thickness of the buffer zone around the device. Must be greater than 0.

    Raises
    ------
    ValueError
        If any of the modes specified in the 'mode' dictionary are not one of the
        allowed values ('constant', 'edge'). Or if the thickness is not greater than 0.

    Example
    -------
        import prefab as pf

        buffer_spec = pf.BufferSpec(
            mode={
                "top": "constant",
                "bottom": "edge",
                "left": "constant",
                "right": "edge",
            },
            thickness=150,
        )
    """

    mode: dict[str, str] = Field(
        default_factory=lambda: {
            "top": "constant",
            "bottom": "constant",
            "left": "constant",
            "right": "constant",
        }
    )
    thickness: conint(gt=0) = 128

    @validator("mode", pre=True)
    def check_mode(cls, v):
        allowed_modes = ["constant", "edge"]
        if not all(mode in allowed_modes for mode in v.values()):
            raise ValueError(f"Buffer mode must be one of {allowed_modes}, got '{v}'")
        return v


class Device(BaseModel):
    device_array: np.ndarray = Field(...)
    buffer_spec: BufferSpec = Field(default_factory=BufferSpec)

    class Config:
        arbitrary_types_allowed = True

    @property
    def shape(self) -> tuple[int, int]:
        return self.device_array.shape

    def __init__(
        self, device_array: np.ndarray, buffer_spec: Optional[BufferSpec] = None
    ):
        """
        Represents the planar geometry of a photonic device design that will have its
        nanofabrication outcome predicted and/or corrected.

        This class is designed to encapsulate the geometric representation of a photonic
        device, facilitating operations such as padding, normalization, binarization,
        ternarization, trimming, and blurring. These operations are useful for preparing
        the device design for prediction or correction. Additionally, the class provides
        methods for exporting the device representation to various formats, including
        ndarray, image files, and GDSII files, supporting a range of analysis and
        fabrication workflows.

        Parameters
        ----------
        device_array : np.ndarray
            A 2D array representing the planar geometry of the device. This array
            undergoes various transformations to predict or correct the nanofabrication
            process.
        buffer_spec : BufferSpec, optional
            Defines the parameters for adding a buffer zone around the device geometry.
            This buffer zone is needed for providing surrounding context for prediction
            or correction and for ensuring seamless integration with the surrounding
            circuitry. By default, a generous padding is applied to accommodate isolated
            structures.

        Attributes
        ----------
        shape : tuple[int, int]
            The shape of the device array.

        Raises
        ------
        ValueError
            If the provided `device_array` is not a numpy ndarray or is not a 2D array,
            indicating an invalid device geometry.
        """
        super().__init__(
            device_array=device_array, buffer_spec=buffer_spec or BufferSpec()
        )
        self._initial_processing()

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def _initial_processing(self):
        buffer_thickness = self.buffer_spec.thickness
        buffer_mode = self.buffer_spec.mode

        self.device_array = np.pad(
            self.device_array,
            pad_width=((buffer_thickness, 0), (0, 0)),
            mode=buffer_mode["top"],
        )
        self.device_array = np.pad(
            self.device_array,
            pad_width=((0, buffer_thickness), (0, 0)),
            mode=buffer_mode["bottom"],
        )
        self.device_array = np.pad(
            self.device_array,
            pad_width=((0, 0), (buffer_thickness, 0)),
            mode=buffer_mode["left"],
        )
        self.device_array = np.pad(
            self.device_array,
            pad_width=((0, 0), (0, buffer_thickness)),
            mode=buffer_mode["right"],
        )

        self.device_array = np.expand_dims(
            self.device_array.astype(np.float32), axis=-1
        )

    @root_validator(pre=True)
    def check_device_array(cls, values):
        device_array = values.get("device_array")
        if not isinstance(device_array, np.ndarray):
            raise ValueError("device_array must be a numpy ndarray.")
        if device_array.ndim != 2:
            raise ValueError("device_array must be a 2D array.")
        return values

    def is_binary(self) -> bool:
        """
        Check if the device geometry is binary.

        Returns
        -------
        bool
            True if the device geometry is binary, False otherwise.
        """
        unique_values = np.unique(self.device_array)
        return (
            np.array_equal(unique_values, [0, 1])
            or np.array_equal(unique_values, [1, 0])
            or np.array_equal(unique_values, [0])
            or np.array_equal(unique_values, [1])
        )

    def _encode_array(self, array):
        image = Image.fromarray(np.uint8(array * 255))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_png = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_png

    def _decode_array(self, encoded_png):
        binary_data = base64.b64decode(encoded_png)
        image = Image.open(io.BytesIO(binary_data))
        return np.array(image) / 255

    def _predict_array(
        self,
        model: Model,
        model_type: str,
        binarize: bool,
    ) -> "Device":
        try:
            with open(os.path.expanduser("~/.prefab.toml")) as file:
                content = file.readlines()
                access_token = None
                refresh_token = None
                for line in content:
                    if "access_token" in line:
                        access_token = line.split("=")[1].strip().strip('"')
                    if "refresh_token" in line:
                        refresh_token = line.split("=")[1].strip().strip('"')
                        break
                if not access_token or not refresh_token:
                    raise ValueError("Token not found in the configuration file.")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not validate user.\n"
                "Please update prefab using: pip install --upgrade prefab.\n"
                "Signup/login and generate a new token.\n"
                "See https://www.prefabphotonics.com/docs/guides/quickstart."
            ) from None
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Refresh-Token": refresh_token,
        }

        predict_data = {
            "device_array": self._encode_array(self.device_array[:, :, 0]),
            "model": model.to_json(),
            "model_type": model_type,
            "binary": binarize,
        }
        json_data = json.dumps(predict_data)

        endpoint_url = "https://prefab-photonics--predict-v1.modal.run"

        with requests.post(
            endpoint_url, data=json_data, headers=headers, stream=True
        ) as response:
            response.raise_for_status()
            event_type = None
            model_descriptions = {"p": "Prediction", "c": "Correction", "s": "SEMulate"}
            progress_bar = tqdm(
                total=100,
                desc=f"{model_descriptions[model_type]}",
                unit="%",
                colour="green",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            )

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("event:"):
                        event_type = decoded_line.split(":")[1].strip()
                    elif decoded_line.startswith("data:"):
                        try:
                            data_content = json.loads(decoded_line.split("data: ")[1])
                            if event_type == "progress":
                                progress = round(100 * data_content["progress"])
                                progress_bar.update(progress - progress_bar.n)
                            elif event_type == "result":
                                results = []
                                for key in sorted(data_content.keys()):
                                    if key.startswith("result"):
                                        decoded_image = self._decode_array(
                                            data_content[key]
                                        )
                                        results.append(decoded_image)

                                if results:
                                    prediction = np.stack(results, axis=-1)
                                    if binarize:
                                        prediction = geometry.binarize_hard(prediction)
                                    progress_bar.close()
                                    return prediction
                            elif event_type == "end":
                                print("Stream ended.")
                                progress_bar.close()
                                break
                            elif event_type == "auth":
                                if "new_refresh_token" in data_content["auth"]:
                                    prefab_file_path = os.path.expanduser(
                                        "~/.prefab.toml"
                                    )
                                    with open(
                                        prefab_file_path, "w", encoding="utf-8"
                                    ) as toml_file:
                                        toml.dump(
                                            {
                                                "access_token": data_content["auth"][
                                                    "new_access_token"
                                                ],
                                                "refresh_token": data_content["auth"][
                                                    "new_refresh_token"
                                                ],
                                            },
                                            toml_file,
                                        )
                            elif event_type == "error":
                                print(f"Error: {data_content['error']}")
                                progress_bar.close()
                        except json.JSONDecodeError:
                            print(
                                "Failed to decode JSON:",
                                decoded_line.split("data: ")[1],
                            )

    def predict(
        self,
        model: Model,
        binarize: bool = False,
    ) -> "Device":
        """
        Predict the nanofabrication outcome of the device using a specified model.

        This method sends the device geometry to a serverless prediction service, which
        uses a specified machine learning model to predict the outcome of the
        nanofabrication process.

        Parameters
        ----------
        model : Model
            The model to use for prediction, representing a specific fabrication process
            and dataset. This model encapsulates details about the fabrication foundry,
            process, material, technology, thickness, and sidewall presence, as defined
            in `models.py`. Each model is associated with a version and dataset that
            detail its creation and the data it was trained on, ensuring the prediction
            is tailored to specific fabrication parameters.
        binarize : bool, optional
            If True, the predicted device geometry will be binarized using a threshold
            method. This is useful for converting probabilistic predictions into binary
            geometries. Defaults to False.

        Returns
        -------
        Device
            A new instance of the Device class with the predicted geometry.

        Raises
        ------
        ValueError
            If the prediction service returns an error or if the response from the
            service cannot be processed correctly.
        """
        prediction_array = self._predict_array(
            model=model,
            model_type="p",
            binarize=binarize,
        )
        return self.model_copy(update={"device_array": prediction_array})

    def correct(
        self,
        model: Model,
        binarize: bool = True,
    ) -> "Device":
        """
        Correct the nanofabrication outcome of the device using a specified model.

        This method sends the device geometry to a serverless correction service, which
        uses a specified machine learning model to correct the outcome of the
        nanofabrication process. The correction aims to adjust the device geometry to
        compensate for known fabrication errors and improve the accuracy of the final
        device structure.

        Parameters
        ----------
        model : Model
            The model to use for correction, representing a specific fabrication process
            and dataset. This model encapsulates details about the fabrication foundry,
            process, material, technology, thickness, and sidewall presence, as defined
            in `models.py`. Each model is associated with a version and dataset that
            detail its creation and the data it was trained on, ensuring the correction
            is tailored to specific fabrication parameters.
        binarize : bool, optional
            If True, the corrected device geometry will be binarized using a threshold
            method. This is useful for converting probabilistic corrections into binary
            geometries. Defaults to True.

        Returns
        -------
        Device
            A new instance of the Device class with the corrected geometry.

        Raises
        ------
        ValueError
            If the correction service returns an error or if the response from the
            service cannot be processed correctly.
        """
        correction_array = self._predict_array(
            model=model,
            model_type="c",
            binarize=binarize,
        )
        return self.model_copy(update={"device_array": correction_array})

    def semulate(
        self,
        model: Model,
    ) -> "Device":
        """
        Simulate the appearance of the device as if viewed under a scanning electron
        microscope (SEM).

        This method applies a specified machine learning model to transform the device
        geometry into a style that resembles an SEM image. This can be useful for
        visualizing how the device might appear under an SEM, which is often used for
        inspecting the surface and composition of materials at high magnification.

        Parameters
        ----------
        model : Model
            The model to use for SEMulation, representing a specific fabrication process
            and dataset. This model encapsulates details about the fabrication foundry,
            process, material, technology, thickness, and sidewall presence, as defined
            in `models.py`. Each model is associated with a version and dataset that
            detail its creation and the data it was trained on, ensuring the SEMulation
            is tailored to specific fabrication parameters.

        Returns
        -------
        Device
            A new instance of the Device class with its geometry transformed to simulate
            an SEM image style.
        """
        semulated_array = self._predict_array(
            model=model,
            model_type="s",
            binarize=False,
        )
        return self.model_copy(update={"device_array": semulated_array})

    def to_ndarray(self) -> np.ndarray:
        """
        Converts the device geometry to an ndarray.

        This method applies the buffer specifications to crop the device array if
        necessary, based on the buffer mode ('edge' or 'constant'). It then returns the
        resulting ndarray representing the device geometry.

        Returns
        -------
        np.ndarray
            The ndarray representation of the device geometry, with any applied buffer
            cropping.
        """
        device_array = np.copy(self.device_array)
        buffer_thickness = self.buffer_spec.thickness
        buffer_mode = self.buffer_spec.mode

        crop_top = buffer_thickness if buffer_mode["top"] == "edge" else 0
        crop_bottom = buffer_thickness if buffer_mode["bottom"] == "edge" else 0
        crop_left = buffer_thickness if buffer_mode["left"] == "edge" else 0
        crop_right = buffer_thickness if buffer_mode["right"] == "edge" else 0

        ndarray = device_array[
            crop_top : device_array.shape[0] - crop_bottom,
            crop_left : device_array.shape[1] - crop_right,
        ]
        return np.squeeze(ndarray)

    def to_img(self, img_path: str = "prefab_device.png"):
        """
        Exports the device geometry as an image file.

        This method converts the device geometry to an ndarray using `to_ndarray`,
        scales the values to the range [0, 255] for image representation, and saves the
        result as an image file.

        Parameters
        ----------
        img_path : str, optional
            The path where the image file will be saved. If not specified, the image is
            saved as "prefab_device.png" in the current directory.
        """
        cv2.imwrite(img_path, 255 * self.flatten().to_ndarray())
        print(f"Saved Device to '{img_path}'")

    def to_gds(
        self,
        gds_path: str = "prefab_device.gds",
        cell_name: str = "prefab_device",
        gds_layer: tuple[int, int] = (1, 0),
        contour_approx_mode: int = 2,
    ):
        """
        Exports the device geometry as a GDSII file.

        This method converts the device geometry into a format suitable for GDSII files.
        The conversion involves contour approximation to simplify the geometry while
        preserving essential features.

        Parameters
        ----------
        gds_path : str, optional
            The path where the GDSII file will be saved. If not specified, the file is
            saved as "prefab_device.gds" in the current directory.
        cell_name : str, optional
            The name of the cell within the GDSII file. If not specified, defaults to
            "prefab_device".
        gds_layer : tuple[int, int], optional
            The layer and datatype to use within the GDSII file. Defaults to (1, 0).
        contour_approx_mode : int, optional
            The mode of contour approximation used during the conversion. Defaults to 2,
            which corresponds to `cv2.CHAIN_APPROX_SIMPLE`, a method that compresses
            horizontal, vertical, and diagonal segments and leaves only their endpoints.
        """
        gdstk_cell = self.flatten()._device_to_gdstk(
            cell_name=cell_name,
            gds_layer=gds_layer,
            contour_approx_mode=contour_approx_mode,
        )
        print(f"Saving GDS to '{gds_path}'...")
        gdstk_library = gdstk.Library()
        gdstk_library.add(gdstk_cell)
        gdstk_library.write_gds(outfile=gds_path, max_points=8190)

    def to_gdstk(
        self,
        cell_name: str = "prefab_device",
        gds_layer: tuple[int, int] = (1, 0),
        contour_approx_mode: int = 2,
    ):
        """
        Converts the device geometry to a GDSTK cell object.

        This method prepares the device geometry for GDSII file export by converting it
        into a GDSTK cell object. GDSTK is a Python module for creating and manipulating
        GDSII layout files. The conversion involves contour approximation to simplify
        the geometry while preserving essential features.

        Parameters
        ----------
        cell_name : str, optional
            The name of the cell to be created. Defaults to "prefab_device".
        gds_layer : tuple[int, int], optional
            The layer and datatype to use within the GDSTK cell. Defaults to (1, 0).
        contour_approx_mode : int, optional
            The mode of contour approximation used during the conversion. Defaults to 2,
            which corresponds to `cv2.CHAIN_APPROX_SIMPLE`, a method that compresses
            horizontal, vertical, and diagonal segments and leaves only their endpoints.

        Returns
        -------
        gdstk.Cell
            The GDSTK cell object representing the device geometry.
        """
        print(f"Creating cell '{cell_name}'...")
        gdstk_cell = self.flatten()._device_to_gdstk(
            cell_name=cell_name,
            gds_layer=gds_layer,
            contour_approx_mode=contour_approx_mode,
        )
        return gdstk_cell

    def _device_to_gdstk(
        self,
        cell_name: str,
        gds_layer: tuple[int, int],
        contour_approx_mode: int,
    ) -> gdstk.Cell:
        approx_mode_mapping = {
            1: cv2.CHAIN_APPROX_NONE,
            2: cv2.CHAIN_APPROX_SIMPLE,
            3: cv2.CHAIN_APPROX_TC89_L1,
            4: cv2.CHAIN_APPROX_TC89_KCOS,
        }

        contours, hierarchy = cv2.findContours(
            np.flipud(self.to_ndarray()).astype(np.uint8),
            cv2.RETR_TREE,
            approx_mode_mapping[contour_approx_mode],
        )

        hierarchy_polygons = {}
        for idx, contour in enumerate(contours):
            level = 0
            current_idx = idx
            while hierarchy[0][current_idx][3] != -1:
                level += 1
                current_idx = hierarchy[0][current_idx][3]

            if len(contour) > 2:
                contour = contour / 1000
                points = [tuple(point) for point in contour.squeeze().tolist()]
                if level not in hierarchy_polygons:
                    hierarchy_polygons[level] = []
                hierarchy_polygons[level].append(points)

        cell = gdstk.Cell(cell_name)
        processed_polygons = []
        for level in sorted(hierarchy_polygons.keys()):
            operation = "or" if level % 2 == 0 else "xor"
            polygons_to_process = hierarchy_polygons[level]

            if polygons_to_process:
                processed_polygons = gdstk.boolean(
                    polygons_to_process,
                    processed_polygons,
                    operation,
                    layer=gds_layer[0],
                    datatype=gds_layer[1],
                )
        for polygon in processed_polygons:
            cell.add(polygon)

        return cell

    def _plot_base(
        self,
        plot_array: np.ndarray,
        show_buffer: bool,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]],
        ax: Optional[Axes],
        **kwargs,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_ylabel("y (nm)")
        ax.set_xlabel("x (nm)")

        min_x, min_y = (0, 0) if bounds is None else bounds[0]
        max_x, max_y = plot_array.shape[::-1] if bounds is None else bounds[1]
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = "end" if max_x == "end" else min(max_x, plot_array.shape[1])
        max_y = "end" if max_y == "end" else min(max_y, plot_array.shape[0])
        max_x = plot_array.shape[1] if max_x == "end" else max_x
        max_y = plot_array.shape[0] if max_y == "end" else max_y
        plot_array = plot_array[
            plot_array.shape[0] - max_y : plot_array.shape[0] - min_y,
            min_x:max_x,
        ]
        extent = [min_x, max_x, min_y, max_y]

        if not np.ma.is_masked(plot_array):
            max_size = (1000, 1000)
            scale_x = min(1, max_size[0] / plot_array.shape[1])
            scale_y = min(1, max_size[1] / plot_array.shape[0])
            fx = min(scale_x, scale_y)
            fy = fx

            plot_array = cv2.resize(
                plot_array,
                dsize=(0, 0),
                fx=fx,
                fy=fy,
                interpolation=cv2.INTER_NEAREST,
            )

        mappable = ax.imshow(
            plot_array,
            extent=extent,
            **kwargs,
        )

        if show_buffer:
            self._add_buffer_visualization(ax)

        return mappable, ax

    def plot(
        self,
        show_buffer: bool = True,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
        level: int = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """
        Visualizes the device geometry.

        This method allows for the visualization of the device geometry. The
        visualization can be customized with various matplotlib parameters and can be
        drawn on an existing matplotlib Axes object or create a new one if none is
        provided.

        Parameters
        ----------
        show_buffer : bool, optional
            If True, visualizes the buffer zones around the device. Defaults to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]], optional
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int, optional
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes], optional
            An existing matplotlib Axes object to draw the device geometry on. If
            None, a new figure and axes will be created. Defaults to None.
        **kwargs
            Additional matplotlib parameters for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object containing the plot. This object can be used for
            further plot customization or saving the plot after the method returns.
        """
        if level is None:
            plot_array = geometry.flatten(self.device_array)[:, :, 0]
        else:
            plot_array = self.device_array[:, :, level]
        _, ax = self._plot_base(
            plot_array=plot_array,
            show_buffer=show_buffer,
            bounds=bounds,
            ax=ax,
            **kwargs,
        )
        return ax

    def plot_contour(
        self,
        linewidth: Optional[int] = None,
        # label: Optional[str] = "Device contour",
        show_buffer: bool = True,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
        level: int = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """
        Visualizes the contour of the device geometry.

        This method plots the contour of the device geometry, emphasizing the edges and
        boundaries of the device. The contour plot can be customized with various
        matplotlib parameters, including line width and color. The plot can be drawn on
        an existing matplotlib Axes object or create a new one if none is provided.

        Parameters
        ----------
        linewidth : Optional[int], optional
            The width of the contour lines. If None, the linewidth is automatically
            determined based on the size of the device array. Defaults to None.
        show_buffer : bool, optional
            If True, the buffer zones around the device will be visualized. By default,
            it is set to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]], optional
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int, optional
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes], optional
            An existing matplotlib Axes object to draw the device contour on. If None, a
            new figure and axes will be created. Defaults to None.
        **kwargs
            Additional matplotlib parameters for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object containing the contour plot. This can be used for
            further customization or saving the plot after the method returns.
        """
        if level is None:
            device_array = geometry.flatten(self.device_array)[:, :, 0]
        else:
            device_array = self.device_array[:, :, level]

        kwargs.setdefault("cmap", "spring")
        if linewidth is None:
            linewidth = device_array.shape[0] // 100

        contours, _ = cv2.findContours(
            geometry.binarize_hard(device_array).astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_array = np.zeros_like(device_array, dtype=np.uint8)
        cv2.drawContours(contour_array, contours, -1, (255,), linewidth)
        contour_array = np.ma.masked_equal(contour_array, 0)

        _, ax = self._plot_base(
            plot_array=contour_array,
            show_buffer=show_buffer,
            bounds=bounds,
            ax=ax,
            **kwargs,
        )
        # cmap = cm.get_cmap(kwargs.get("cmap", "spring"))
        # legend_proxy = Line2D([0], [0], linestyle="-", color=cmap(1))
        # ax.legend([legend_proxy], [label], loc="upper right")
        return ax

    def plot_uncertainty(
        self,
        show_buffer: bool = True,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
        level: int = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """
        Visualizes the uncertainty in the edge positions of the predicted device.

        This method plots the uncertainty associated with the positions of the edges of
        the device. The uncertainty is represented as a gradient, with areas of higher
        uncertainty indicating a greater likelihood of the edge position from run to run
        (due to inconsistencies in the fabrication process). This visualization can help
        in identifying areas within the device geometry that may require design
        adjustments to improve fabrication consistency.

        Parameters
        ----------
        show_buffer : bool, optional
            If True, the buffer zones around the device will also be visualized. By
            default, it is set to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]], optional
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int, optional
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes], optional
            An existing matplotlib Axes object to draw the uncertainty visualization on.
            If None, a new figure and axes will be created. Defaults to None.
        **kwargs
            Additional matplotlib parameters for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object containing the uncertainty visualization. This
            can be used for further customization or saving the plot after the method
            returns.
        """
        uncertainty_array = self.get_uncertainty()

        if level is None:
            uncertainty_array = geometry.flatten(uncertainty_array)[:, :, 0]
        else:
            uncertainty_array = uncertainty_array[:, :, level]

        mappable, ax = self._plot_base(
            plot_array=uncertainty_array,
            show_buffer=show_buffer,
            bounds=bounds,
            ax=ax,
            **kwargs,
        )
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label("Uncertainty (a.u.)")
        return ax

    def plot_compare(
        self,
        ref_device: "Device",
        show_buffer: bool = True,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
        level: int = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """
        Visualizes the comparison between the current device geometry and a reference
        device geometry.

        Positive values (dilation) and negative values (erosion) are visualized with a
        color map to indicate areas where the current device has expanded or contracted
        relative to the reference.

        Parameters
        ----------
        ref_device : Device
            The reference device to compare against.
        show_buffer : bool, optional
            If True, visualizes the buffer zones around the device. Defaults to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]], optional
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int, optional
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes], optional
            An existing matplotlib Axes object to draw the comparison on. If None, a new
            figure and axes will be created. Defaults to None.
        **kwargs
            Additional matplotlib parameters for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object containing the comparison plot. This object can
            be used for further plot customization or saving the plot after the method
            returns.
        """
        plot_array = ref_device.device_array - self.device_array

        if level is None:
            plot_array = geometry.flatten(plot_array)[:, :, 0]
        else:
            plot_array = plot_array[:, :, level]

        mappable, ax = self._plot_base(
            plot_array=plot_array,
            show_buffer=show_buffer,
            bounds=bounds,
            ax=ax,
            cmap="jet",
            **kwargs,
        )
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label("Added (a.u.)                        Removed (a.u.)")
        return ax

    def _add_buffer_visualization(self, ax: Axes):
        plot_array = self.device_array

        buffer_thickness = self.buffer_spec.thickness
        buffer_fill = (0, 1, 0, 0.2)
        buffer_hatch = "/"

        mid_rect = Rectangle(
            (buffer_thickness, buffer_thickness),
            plot_array.shape[1] - 2 * buffer_thickness,
            plot_array.shape[0] - 2 * buffer_thickness,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(mid_rect)

        top_rect = Rectangle(
            (0, 0),
            plot_array.shape[1],
            buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(top_rect)

        bottom_rect = Rectangle(
            (0, plot_array.shape[0] - buffer_thickness),
            plot_array.shape[1],
            buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(bottom_rect)

        left_rect = Rectangle(
            (0, buffer_thickness),
            buffer_thickness,
            plot_array.shape[0] - 2 * buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(left_rect)

        right_rect = Rectangle(
            (
                plot_array.shape[1] - buffer_thickness,
                buffer_thickness,
            ),
            buffer_thickness,
            plot_array.shape[0] - 2 * buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(right_rect)

    def normalize(self) -> "Device":
        """
        Normalize the device geometry.

        Returns
        -------
        Device
            A new instance of the Device with the normalized geometry.
        """
        normalized_device_array = geometry.normalize(device_array=self.device_array)
        return self.model_copy(update={"device_array": normalized_device_array})

    def binarize(self, eta: float = 0.5, beta: float = np.inf) -> "Device":
        """
        Binarize the device geometry based on a threshold and a scaling factor.

        Parameters
        ----------
        eta : float, optional
            The threshold value for binarization. Defaults to 0.5.
        beta : float, optional
            The scaling factor for the binarization process. A higher value makes the
            transition sharper. Defaults to np.inf, which results in a hard threshold.

        Returns
        -------
        Device
            A new instance of the Device with the binarized geometry.
        """
        binarized_device_array = geometry.binarize(
            device_array=self.device_array, eta=eta, beta=beta
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def binarize_hard(self, eta: float = 0.5) -> "Device":
        """
        Apply a hard threshold to binarize the device geometry. The `binarize` function
        is generally preferred for most use cases, but it can create numerical artifacts
        for large beta values.

            Parameters
            ----------
            eta : float, optional
                The threshold value for binarization. Defaults to 0.5.

            Returns
            -------
            Device
                A new instance of the Device with the threshold-binarized geometry.
        """
        binarized_device_array = geometry.binarize_hard(
            device_array=self.device_array, eta=eta
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def binarize_monte_carlo(
        self,
        threshold_noise_std: float = 2.0,
        threshold_blur_std: float = 9.0,
    ) -> "Device":
        """
        Binarize the device geometry using a Monte Carlo approach with Gaussian
        blurring.

        This method applies a dynamic thresholding technique where the threshold value
        is determined by a base value perturbed by Gaussian-distributed random noise.
        The threshold is then spatially varied across the device array using Gaussian
        blurring, simulating a more realistic scenario where the threshold is not
        uniform across the device.

        Parameters
        ----------
        threshold_noise_std : float, optional
            The standard deviation of the Gaussian distribution used to generate noise
            for the threshold values. This controls the amount of randomness in the
            threshold. Defaults to 2.0.
        threshold_blur_std : float, optional
            The standard deviation for the Gaussian kernel used in blurring the
            threshold map. This controls the spatial variation of the threshold across
            the array. Defaults to 9.0.

        Returns
        -------
        Device
            A new instance of the Device with the binarized geometry.
        """
        binarized_device_array = geometry.binarize_monte_carlo(
            device_array=self.device_array,
            threshold_noise_std=threshold_noise_std,
            threshold_blur_std=threshold_blur_std,
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def ternarize(self, eta1: float = 1 / 3, eta2: float = 2 / 3) -> "Device":
        """
        Ternarize the device geometry based on two thresholds. This function is useful
        for flattened devices with angled sidewalls (i.e., three segments).

        Parameters
        ----------
        eta1 : float, optional
            The first threshold value for ternarization. Defaults to 1/3.
        eta2 : float, optional
            The second threshold value for ternarization. Defaults to 2/3.

        Returns
        -------
        Device
            A new instance of the Device with the ternarized geometry.
        """
        ternarized_device_array = geometry.ternarize(
            device_array=self.flatten().device_array, eta1=eta1, eta2=eta2
        )
        return self.model_copy(update={"device_array": ternarized_device_array})

    def trim(self) -> "Device":
        """
        Trim the device geometry by removing empty space around it.

        Parameters
        ----------
        buffer_thickness : int, optional
            The thickness of the buffer to leave around the empty space. Defaults to 0,
            which means no buffer is added.

        Returns
        -------
        Device
            A new instance of the Device with the trimmed geometry.
        """
        trimmed_device_array = geometry.trim(
            device_array=self.device_array,
            buffer_thickness=self.buffer_spec.thickness,
        )
        return self.model_copy(update={"device_array": trimmed_device_array})

    def blur(self, sigma: float = 1.0) -> "Device":
        """
        Apply Gaussian blur to the device geometry and normalize the result.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation for the Gaussian kernel. This controls the amount of
            blurring. Defaults to 1.0.

        Returns
        -------
        Device
            A new instance of the Device with the blurred and normalized geometry.
        """
        blurred_device_array = geometry.blur(
            device_array=self.device_array, sigma=sigma
        )
        return self.model_copy(update={"device_array": blurred_device_array})

    def rotate(self, angle: float) -> "Device":
        """
        Rotate the device geometry by a given angle.

        Parameters
        ----------
        angle : float
            The angle of rotation in degrees. Positive values mean counter-clockwise
            rotation.

        Returns
        -------
        Device
            A new instance of the Device with the rotated geometry.
        """
        rotated_device_array = geometry.rotate(
            device_array=self.device_array, angle=angle
        )
        return self.model_copy(update={"device_array": rotated_device_array})

    def erode(self, kernel_size: int = 3) -> "Device":
        """
        Erode the device geometry by removing small areas of overlap.

        Parameters
        ----------
        kernel_size : int
            The size of the kernel used for erosion.

        Returns
        -------
        Device
            A new instance of the Device with the eroded geometry.
        """
        eroded_device_array = geometry.erode(
            device_array=self.device_array, kernel_size=kernel_size
        )
        return self.model_copy(update={"device_array": eroded_device_array})

    def dilate(self, kernel_size: int = 3) -> "Device":
        """
        Dilate the device geometry by expanding areas of overlap.

        Parameters
        ----------
        kernel_size : int
            The size of the kernel used for dilation.

        Returns
        -------
        Device
            A new instance of the Device with the dilated geometry.
        """
        dilated_device_array = geometry.dilate(
            device_array=self.device_array, kernel_size=kernel_size
        )
        return self.model_copy(update={"device_array": dilated_device_array})

    def flatten(self) -> "Device":
        """
        Flatten the device geometry by summing the vertical layers and normalizing the
        result.

        Parameters
        ----------
        device_array : np.ndarray
            The input array to be flattened.

        Returns
        -------
        np.ndarray
            The flattened array with values scaled between 0 and 1.
        """
        flattened_device_array = geometry.flatten(device_array=self.device_array)
        return self.model_copy(update={"device_array": flattened_device_array})

    def get_uncertainty(self) -> np.ndarray:
        """
        Calculate the uncertainty in the edge positions of the predicted device.

        This method computes the uncertainty based on the deviation of the device's
        geometry values from the midpoint (0.5). The uncertainty is defined as the
        absolute difference from 0.5, scaled and inverted to provide a measure where
        higher values indicate greater uncertainty.

        Returns
        -------
        np.ndarray
            An array representing the uncertainty in the edge positions of the device,
            with higher values indicating greater uncertainty.
        """
        return 1 - 2 * np.abs(0.5 - self.device_array)
