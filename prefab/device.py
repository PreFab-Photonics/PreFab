"""Provides the Device class for representing photonic devices."""

from typing import TYPE_CHECKING, Literal, Optional

import cv2
import gdstk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import BaseModel, Field, model_validator, validator
from scipy.ndimage import distance_transform_edt
from skimage import measure

from . import compare, geometry
from .models import Model
from .predict import predict_array

if TYPE_CHECKING:
    import gdsfactory as gf
    import tidy3d as td

Image.MAX_IMAGE_PIXELS = None


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
        ('top', 'bottom', 'left', 'right'), where:
        - 'constant' is used for isolated structures
        - 'edge' is utilized for preserving the edge, such as for waveguide connections
        - 'none' for no buffer on that side
    thickness : dict[str, int]
        A dictionary that defines the thickness of the buffer zone for each side of the
        device ('top', 'bottom', 'left', 'right'). Each value must be greater than or
        equal to 0.

    Raises
    ------
    ValueError
        If any of the modes specified in the 'mode' dictionary are not one of the
        allowed values ('constant', 'edge', 'none'). Or if any of the thickness values
        are negative.

    Example
    -------
        import prefab as pf

        buffer_spec = pf.BufferSpec(
            mode={
                "top": "constant",
                "bottom": "none",
                "left": "constant",
                "right": "edge",
            },
            thickness={
                "top": 150,
                "bottom": 0,
                "left": 200,
                "right": 250,
            },
        )
    """

    mode: dict[str, Literal["constant", "edge", "none"]] = Field(
        default_factory=lambda: {
            "top": "constant",
            "bottom": "constant",
            "left": "constant",
            "right": "constant",
        }
    )
    thickness: dict[str, int] = Field(
        default_factory=lambda: {
            "top": 128,
            "bottom": 128,
            "left": 128,
            "right": 128,
        }
    )

    @validator("mode", pre=True)
    def check_mode(cls, v):
        allowed_modes = ["constant", "edge", "none"]
        if not all(mode in allowed_modes for mode in v.values()):
            raise ValueError(f"Buffer mode must be one of {allowed_modes}, got '{v}'.")
        return v

    @validator("thickness")
    def check_thickness(cls, v):
        if not all(t >= 0 for t in v.values()):
            raise ValueError("All thickness values must be greater than or equal to 0.")
        return v

    @model_validator(mode="after")
    def check_none_thickness(cls, values):
        mode = values.mode
        thickness = values.thickness
        for side in mode:
            if mode[side] == "none" and thickness[side] != 0:
                raise ValueError(
                    f"Thickness must be 0 when mode is 'none' for {side} side"
                )
            if mode[side] != "none" and thickness[side] == 0:
                raise ValueError(
                    f"Mode must be 'none' when thickness is 0 for {side} side"
                )
        return values


class Device(BaseModel):
    device_array: np.ndarray = Field(...)
    buffer_spec: BufferSpec = Field(default_factory=BufferSpec)

    class Config:
        arbitrary_types_allowed = True

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.device_array.shape)

    def __init__(
        self, device_array: np.ndarray, buffer_spec: Optional[BufferSpec] = None
    ):
        """
        Represents the planar geometry of a photonic device design that will have its
        nanofabrication outcome predicted and/or corrected.

        This class is designed to encapsulate the geometric representation of a photonic
        device, facilitating operations such as padding, normalization, binarization,
        erosion/dilation, trimming, and blurring. These operations are useful for
        preparingthe device design for prediction or correction. Additionally, the class
        providesmethods for exporting the device representation to various formats,
        includingndarray, image files, and GDSII files, supporting a range of analysis
        and fabrication workflows.

        Parameters
        ----------
        device_array : np.ndarray
            A 2D array representing the planar geometry of the device. This array
            undergoes various transformations to predict or correct the nanofabrication
            process.
        buffer_spec : Optional[BufferSpec]
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

        if buffer_mode["top"] != "none":
            self.device_array = np.pad(
                self.device_array,
                pad_width=((buffer_thickness["top"], 0), (0, 0)),
                mode=buffer_mode["top"],
            )

        if buffer_mode["bottom"] != "none":
            self.device_array = np.pad(
                self.device_array,
                pad_width=((0, buffer_thickness["bottom"]), (0, 0)),
                mode=buffer_mode["bottom"],
            )

        if buffer_mode["left"] != "none":
            self.device_array = np.pad(
                self.device_array,
                pad_width=((0, 0), (buffer_thickness["left"], 0)),
                mode=buffer_mode["left"],
            )

        if buffer_mode["right"] != "none":
            self.device_array = np.pad(
                self.device_array,
                pad_width=((0, 0), (0, buffer_thickness["right"])),
                mode=buffer_mode["right"],
            )

        self.device_array = np.expand_dims(self.device_array, axis=-1)

    @model_validator(mode="before")
    def check_device_array(cls, values):
        device_array = values.get("device_array")
        if not isinstance(device_array, np.ndarray):
            raise ValueError("device_array must be a numpy ndarray.")
        if device_array.ndim != 2:
            raise ValueError("device_array must be a 2D array.")
        return values

    @property
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

    def predict(
        self,
        model: Model,
        binarize: bool = False,
        gpu: bool = False,
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
        binarize : bool
            If True, the predicted device geometry will be binarized using a threshold
            method. This is useful for converting probabilistic predictions into binary
            geometries. Defaults to False.
        gpu : bool
            If True, the prediction will be performed on a GPU. Defaults to False.
            Note: The GPU option has more overhead and will take longer for small
            devices, but will be faster for larger devices.

        Returns
        -------
        Device
            A new instance of the Device class with the predicted geometry.

        Raises
        ------
        RuntimeError
            If the prediction service returns an error or if the response from the
            service cannot be processed correctly.
        """
        prediction_array = predict_array(
            device_array=self.device_array,
            model=model,
            model_type="p",
            binarize=binarize,
            gpu=gpu,
        )
        return self.model_copy(update={"device_array": prediction_array})

    def correct(
        self,
        model: Model,
        binarize: bool = True,
        gpu: bool = False,
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
        binarize : bool
            If True, the corrected device geometry will be binarized using a threshold
            method. This is useful for converting probabilistic corrections into binary
            geometries. Defaults to True.
        gpu : bool
            If True, the prediction will be performed on a GPU. Defaults to False.
            Note: The GPU option has more overhead and will take longer for small
            devices, but will be faster for larger devices.

        Returns
        -------
        Device
            A new instance of the Device class with the corrected geometry.

        Raises
        ------
        RuntimeError
            If the correction service returns an error or if the response from the
            service cannot be processed correctly.
        """
        correction_array = predict_array(
            device_array=self.device_array,
            model=model,
            model_type="c",
            binarize=binarize,
            gpu=gpu,
        )
        return self.model_copy(update={"device_array": correction_array})

    def semulate(
        self,
        model: Model,
        gpu: bool = False,
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
        gpu : bool
            If True, the prediction will be performed on a GPU. Defaults to False.
            Note: The GPU option has more overhead and will take longer for small
            devices, but will be faster for larger devices.

        Notes
        -----
        The salt-and-pepper noise is added manually until the model is trained to
        generate this noise (not a big priority).

        Returns
        -------
        Device
            A new instance of the Device class with its geometry transformed to simulate
            an SEM image style.

        Raises
        ------
        RuntimeError
            If the prediction service returns an error or if the response from the
            service cannot be processed correctly.
        """
        semulated_array = predict_array(
            device_array=self.device_array,
            model=model,
            model_type="s",
            binarize=False,
            gpu=gpu,
        )
        semulated_array += np.random.normal(0, 0.03, semulated_array.shape)
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

        crop_top = buffer_thickness["top"] if buffer_mode["top"] == "constant" else 0
        crop_bottom = (
            buffer_thickness["bottom"] if buffer_mode["bottom"] == "constant" else 0
        )
        crop_left = buffer_thickness["left"] if buffer_mode["left"] == "constant" else 0
        crop_right = (
            buffer_thickness["right"] if buffer_mode["right"] == "constant" else 0
        )

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
        img_path : str
            The path where the image file will be saved. If not specified, the image is
            saved as "prefab_device.png" in the current directory.
        """
        cv2.imwrite(img_path, 255 * self.flatten().to_ndarray())
        print(f"Saved Device image to '{img_path}'")

    def to_gds(
        self,
        gds_path: str = "prefab_device.gds",
        cell_name: str = "prefab_device",
        gds_layer: tuple[int, int] = (1, 0),
        contour_approx_mode: int = 2,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        """
        Exports the device geometry as a GDSII file.

        This method converts the device geometry into a format suitable for GDSII files.
        The conversion involves contour approximation to simplify the geometry while
        preserving essential features.

        Parameters
        ----------
        gds_path : str
            The path where the GDSII file will be saved. If not specified, the file is
            saved as "prefab_device.gds" in the current directory.
        cell_name : str
            The name of the cell within the GDSII file. If not specified, defaults to
            "prefab_device".
        gds_layer : tuple[int, int]
            The layer and datatype to use within the GDSII file. Defaults to (1, 0).
        contour_approx_mode : int
            The mode of contour approximation used during the conversion. Defaults to 2,
            which corresponds to `cv2.CHAIN_APPROX_SIMPLE`, a method that compresses
            horizontal, vertical, and diagonal segments and leaves only their endpoints.
        origin : tuple[float, float]
            The x and y coordinates of the origin in µm for the GDSII export. Defaults
            to (0.0, 0.0).
        """
        gdstk_cell = self.flatten()._device_to_gdstk(
            cell_name=cell_name,
            gds_layer=gds_layer,
            contour_approx_mode=contour_approx_mode,
            origin=origin,
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
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        """
        Converts the device geometry to a GDSTK cell object.

        This method prepares the device geometry for GDSII file export by converting it
        into a GDSTK cell object. GDSTK is a Python module for creating and manipulating
        GDSII layout files. The conversion involves contour approximation to simplify
        the geometry while preserving essential features.

        Parameters
        ----------
        cell_name : str
            The name of the cell to be created. Defaults to "prefab_device".
        gds_layer : tuple[int, int]
            The layer and datatype to use within the GDSTK cell. Defaults to (1, 0).
        contour_approx_mode : int
            The mode of contour approximation used during the conversion. Defaults to 2,
            which corresponds to `cv2.CHAIN_APPROX_SIMPLE`, a method that compresses
            horizontal, vertical, and diagonal segments and leaves only their endpoints.
        origin : tuple[float, float]
            The x and y coordinates of the origin in µm for the GDSTK cell. Defaults
            to (0.0, 0.0).

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
            origin=origin,
        )
        return gdstk_cell

    def _device_to_gdstk(
        self,
        cell_name: str,
        gds_layer: tuple[int, int],
        contour_approx_mode: int,
        origin: tuple[float, float],
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
                buffer_thickness = self.buffer_spec.thickness

                center_x_nm = (
                    self.device_array.shape[1]
                    - buffer_thickness["left"]
                    - buffer_thickness["right"]
                ) / 2
                center_y_nm = (
                    self.device_array.shape[0]
                    - buffer_thickness["top"]
                    - buffer_thickness["bottom"]
                ) / 2

                center_x_um = center_x_nm / 1000
                center_y_um = center_y_nm / 1000

                adjusted_polygons = [
                    gdstk.Polygon(
                        [
                            (x - center_x_um + origin[0], y - center_y_um + origin[1])
                            for x, y in polygon
                        ]
                    )
                    for polygon in polygons_to_process
                ]
                processed_polygons = gdstk.boolean(
                    adjusted_polygons,
                    processed_polygons,
                    operation,
                    layer=gds_layer[0],
                    datatype=gds_layer[1],
                )
        for polygon in processed_polygons:
            cell.add(polygon)

        return cell

    def to_gdsfactory(self) -> "gf.Component":
        """
        Convert the device geometry to a gdsfactory Component.

        Returns
        -------
        gf.Component
            A gdsfactory Component object representing the device geometry.

        Raises
        ------
        ImportError
            If the gdsfactory package is not installed.
        """
        try:
            import gdsfactory as gf
        except ImportError:
            raise ImportError(
                "The gdsfactory package is required to use this function; "
                "try `pip install gdsfactory`."
            ) from None

        device_array = np.rot90(self.to_ndarray(), k=-1)
        return gf.read.from_np(device_array, nm_per_pixel=1)

    def to_tidy3d(
        self,
        eps0: float,
        thickness: float,
    ) -> "td.Structure":
        """
        Convert the device geometry to a Tidy3D Structure.

        Parameters
        ----------
        eps0 : float
            The permittivity value to assign to the device array.
        thickness : float
            The thickness of the device in the z-direction.

        Returns
        -------
        td.Structure
            A Tidy3D Structure object representing the device geometry.

        Raises
        ------
        ImportError
            If the tidy3d package is not installed.
        """
        try:
            from tidy3d import Box, CustomMedium, SpatialDataArray, Structure, inf
        except ImportError:
            raise ImportError(
                "The tidy3d package is required to use this function; "
                "try `pip install tidy3d`."
            ) from None

        X = np.linspace(-self.shape[1] / 2000, self.shape[1] / 2000, self.shape[1])
        Y = np.linspace(-self.shape[0] / 2000, self.shape[0] / 2000, self.shape[0])
        Z = np.array([0])

        device_array = np.rot90(np.fliplr(self.device_array), k=1)
        eps_array = np.where(device_array >= 1.0, eps0, device_array)
        eps_array = np.where(eps_array < 1.0, 1.0, eps_array)
        eps_dataset = SpatialDataArray(eps_array, coords=dict(x=X, y=Y, z=Z))
        medium = CustomMedium.from_eps_raw(eps_dataset)
        return Structure(
            geometry=Box(center=(0, 0, 0), size=(inf, inf, thickness), attrs={}),
            medium=medium,
            name="device",
            attrs={},
        )

    def to_3d(self, thickness_nm: int) -> np.ndarray:
        """
        Convert the 2D device geometry into a 3D representation.

        This method creates a 3D array by interpolating between the bottom and top
        layers of the device geometry. The interpolation is linear.

        Parameters
        ----------
        thickness_nm : int
            The thickness of the 3D representation in nanometers.

        Returns
        -------
        np.ndarray
            A 3D narray representing the device geometry with the specified thickness.
        """
        bottom_layer = self.device_array[:, :, 0]
        top_layer = self.device_array[:, :, -1]
        dt_bottom = np.array(distance_transform_edt(bottom_layer)) - np.array(
            distance_transform_edt(1 - bottom_layer)
        )
        dt_top = np.array(distance_transform_edt(top_layer)) - np.array(
            distance_transform_edt(1 - top_layer)
        )
        weights = np.linspace(0, 1, thickness_nm)
        layered_array = np.zeros(
            (bottom_layer.shape[0], bottom_layer.shape[1], thickness_nm)
        )
        for i, w in enumerate(weights):
            dt_interp = (1 - w) * dt_bottom + w * dt_top
            layered_array[:, :, i] = dt_interp >= 0
        return layered_array

    def to_stl(self, thickness_nm: int, filename: str = "prefab_device.stl"):
        """
        Export the device geometry as an STL file.

        Parameters
        ----------
        thickness_nm : int
            The thickness of the 3D representation in nanometers.
        filename : str
            The name of the STL file to save. Defaults to "prefab_device.stl".

        Raises
        ------
        ValueError
            If the thickness is not a positive integer.
        ImportError
            If the numpy-stl package is not installed.
        """
        try:
            from stl import mesh  # type: ignore
        except ImportError:
            raise ImportError(
                "The stl package is required to use this function; "
                "try `pip install numpy-stl`."
            ) from None

        if thickness_nm <= 0:
            raise ValueError("Thickness must be a positive integer.")

        layered_array = self.to_3d(thickness_nm)
        layered_array = np.pad(
            layered_array, ((0, 0), (0, 0), (10, 10)), mode="constant"
        )
        verts, faces, _, _ = measure.marching_cubes(layered_array, level=0.5)
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = verts[f[j], :]
        cube.save(filename)
        print(f"Saved Device to '{filename}'")

    def _plot_base(
        self,
        plot_array: np.ndarray,
        show_buffer: bool,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]],
        ax: Optional[Axes],
        **kwargs,
    ) -> tuple[plt.cm.ScalarMappable, Axes]:
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
            extent=(
                float(min_x),
                float(max_x),
                float(min_y),
                float(max_y),
            ),
            **kwargs,
        )

        if show_buffer:
            self._add_buffer_visualization(ax)

        # # Adjust colorbar font size if a colorbar is added
        # if "cmap" in kwargs:
        #     cbar = plt.colorbar(mappable, ax=ax)
        #     cbar.ax.tick_params(labelsize=14)
        #     if "label" in kwargs:
        #         cbar.set_label(kwargs["label"], fontsize=16)

        return mappable, ax

    def plot(
        self,
        show_buffer: bool = True,
        bounds: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
        level: Optional[int] = None,
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
        show_buffer : bool
            If True, visualizes the buffer zones around the device. Defaults to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]], optional
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes]
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
        level: Optional[int] = None,
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
        linewidth : Optional[int]
            The width of the contour lines. If None, the linewidth is automatically
            determined based on the size of the device array. Defaults to None.
        show_buffer : bool
            If True, the buffer zones around the device will be visualized. By default,
            it is set to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]]
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes]
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
        level: Optional[int] = None,
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
        show_buffer : bool
            If True, the buffer zones around the device will also be visualized. By
            default, it is set to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]]
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes]
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
        level: Optional[int] = None,
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
        show_buffer : bool
            If True, visualizes the buffer zones around the device. Defaults to True.
        bounds : Optional[tuple[tuple[int, int], tuple[int, int]]]
            Specifies the bounds for zooming into the device geometry, formatted as
            ((min_x, min_y), (max_x, max_y)). If 'max_x' or 'max_y' is set to "end", it
            will be replaced with the corresponding dimension size of the device array.
            If None, the entire device geometry is visualized.
        level : int
            The vertical layer to plot. If None, the device geometry is flattened.
            Defaults to None.
        ax : Optional[Axes]
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
            (buffer_thickness["left"], buffer_thickness["top"]),
            plot_array.shape[1] - buffer_thickness["left"] - buffer_thickness["right"],
            plot_array.shape[0] - buffer_thickness["top"] - buffer_thickness["bottom"],
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(mid_rect)

        top_rect = Rectangle(
            (0, 0),
            plot_array.shape[1],
            buffer_thickness["top"],
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(top_rect)

        bottom_rect = Rectangle(
            (0, plot_array.shape[0] - buffer_thickness["bottom"]),
            plot_array.shape[1],
            buffer_thickness["bottom"],
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(bottom_rect)

        left_rect = Rectangle(
            (0, buffer_thickness["top"]),
            buffer_thickness["left"],
            plot_array.shape[0] - buffer_thickness["top"] - buffer_thickness["bottom"],
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(left_rect)

        right_rect = Rectangle(
            (
                plot_array.shape[1] - buffer_thickness["right"],
                buffer_thickness["top"],
            ),
            buffer_thickness["right"],
            plot_array.shape[0] - buffer_thickness["top"] - buffer_thickness["bottom"],
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
        eta : float
            The threshold value for binarization. Defaults to 0.5.
        beta : float
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
        return self.model_copy(
            update={"device_array": binarized_device_array.astype(np.uint8)}
        )

    def binarize_hard(self, eta: float = 0.5) -> "Device":
        """
        Apply a hard threshold to binarize the device geometry. The `binarize` function
        is generally preferred for most use cases, but it can create numerical artifacts
        for large beta values.

        Parameters
        ----------
        eta : float
            The threshold value for binarization. Defaults to 0.5.

        Returns
        -------
        Device
            A new instance of the Device with the threshold-binarized geometry.
        """
        binarized_device_array = geometry.binarize_hard(
            device_array=self.device_array, eta=eta
        )
        return self.model_copy(
            update={"device_array": binarized_device_array.astype(np.uint8)}
        )

    def binarize_monte_carlo(
        self,
        noise_magnitude: float = 2.0,
        blur_radius: float = 8.0,
    ) -> "Device":
        """
        Binarize the input ndarray using a dynamic thresholding approach to simulate
        surfaceroughness.

        This function applies a dynamic thresholding technique where the threshold value
        is determined by a base value perturbed by Gaussian-distributed random noise.
        Thethreshold is then spatially varied across the array using Gaussian blurring,
        simulating a potentially more realistic scenario where the threshold is not
        uniform across the device.

        Notes
        -----
        This is a temporary solution, where the defaults are chosen based on what looks
        good. A better, data-driven approach is needed.

        Parameters
        ----------
        noise_magnitude : float
            The standard deviation of the Gaussian distribution used to generate noise
            for the threshold values. This controls the amount of randomness in the
            threshold. Defaults to 2.0.
        blur_radius : float
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
            noise_magnitude=noise_magnitude,
            blur_radius=blur_radius,
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def ternarize(self, eta1: float = 1 / 3, eta2: float = 2 / 3) -> "Device":
        """
        Ternarize the device geometry based on two thresholds. This function is useful
        for flattened devices with angled sidewalls (i.e., three segments).

        Parameters
        ----------
        eta1 : float
            The first threshold value for ternarization. Defaults to 1/3.
        eta2 : float
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

    def pad(self, pad_width: int) -> "Device":
        """
        Pad the device geometry with a specified width on all sides.
        """
        padded_device_array = geometry.pad(
            device_array=self.device_array, pad_width=pad_width
        )
        return self.model_copy(update={"device_array": padded_device_array})

    def blur(self, sigma: float = 1.0) -> "Device":
        """
        Apply Gaussian blur to the device geometry and normalize the result.

        Parameters
        ----------
        sigma : float
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

        Returns
        -------
        Device
            A new instance of the Device with the flattened geometry.
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

    def enforce_feature_size(
        self, min_feature_size: int, strel: str = "disk"
    ) -> "Device":
        """
        Enforce a minimum feature size on the device geometry.

        This method applies morphological operations to ensure that all features in the
        device geometry are at least the specified minimum size. It uses either a disk
        or square structuring element for the operations.

        Notes
        -----
        This function does not guarantee that the minimum feature size is enforced in
        all cases. A better process is needed.

        Parameters
        ----------
        min_feature_size : int
            The minimum feature size to enforce, in nanometers.
        strel : str
            The type of structuring element to use. Can be either "disk" or "square".
            Defaults to "disk".

        Returns
        -------
        Device
            A new instance of the Device with the modified geometry.

        Raises
        ------
        ValueError
            If an invalid structuring element type is specified.
        """
        modified_geometry = geometry.enforce_feature_size(
            device_array=self.device_array,
            min_feature_size=min_feature_size,
            strel=strel,
        )
        return self.model_copy(update={"device_array": modified_geometry})

    def check_feature_size(self, min_feature_size: int, strel: str = "disk"):
        """
        Check and visualize the effect of enforcing a minimum feature size on the device
        geometry.

        This method enforces a minimum feature size on the device geometry using the
        specified structuring element, compares the modified geometry with the original,
        and plots the differences. It also calculates and prints the Hamming distance
        between the original and modified geometries, providing a measure of the changes
        introduced by the feature size enforcement.

        Notes
        -----
        This is not a design-rule-checking function, but it can be useful for quick
        checks.

        Parameters
        ----------
        min_feature_size : int
            The minimum feature size to enforce, in nanometers.
        strel : str
            The type of structuring element to use. Can be either "disk" or "square".
            Defaults to "disk".

        Raises
        ------
        ValueError
            If an invalid structuring element type is specified or if min_feature_size
            is not a positive integer.
        """
        if min_feature_size <= 0:
            raise ValueError("min_feature_size must be a positive integer.")

        enforced_device = self.enforce_feature_size(min_feature_size, strel)

        difference = np.abs(
            enforced_device.device_array[:, :, 0] - self.device_array[:, :, 0]
        )
        _, ax = self._plot_base(
            plot_array=difference,
            show_buffer=False,
            ax=None,
            bounds=None,
            cmap="jet",
        )

        hamming_distance = compare.hamming_distance(self, enforced_device)
        print(
            f"Feature size check with minimum size {min_feature_size} "
            f"using '{strel}' structuring element resulted in a Hamming "
            f"distance of: {hamming_distance}"
        )
