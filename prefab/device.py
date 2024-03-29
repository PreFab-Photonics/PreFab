from typing import Optional

import cv2
import gdstk
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pydantic import BaseModel, Field, conint, root_validator, validator

from prefab import geometry


class BufferSpec(BaseModel):
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
    device_array: np.ndarray
    buffer_spec: BufferSpec = Field(default_factory=BufferSpec)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initial_processing()

    def _initial_processing(self):
        buffer_thickness = self.buffer_spec.thickness
        buffer_mode = self.buffer_spec.mode

        self.device_array = np.pad(
            self.device_array,
            ((buffer_thickness, 0), (0, 0)),
            mode=buffer_mode["top"],
        )
        self.device_array = np.pad(
            self.device_array,
            ((0, buffer_thickness), (0, 0)),
            mode=buffer_mode["bottom"],
        )
        self.device_array = np.pad(
            self.device_array,
            ((0, 0), (buffer_thickness, 0)),
            mode=buffer_mode["left"],
        )
        self.device_array = np.pad(
            self.device_array,
            ((0, 0), (0, buffer_thickness)),
            mode=buffer_mode["right"],
        )

        self.device_array = self.device_array.astype(np.float32)

    @root_validator(pre=True)
    def check_device_array(cls, values):
        device_array = values.get("device_array")
        if device_array is not None:
            if not isinstance(device_array, np.ndarray):
                raise ValueError("device_array must be a numpy ndarray.")
            if device_array.ndim != 2:
                raise ValueError("device_array must be a 2D array.")
        return values

    def to_ndarray(self) -> np.ndarray:
        device_array = np.copy(self.device_array)
        buffer_thickness = self.buffer_spec.thickness
        buffer_mode = self.buffer_spec.mode

        if buffer_mode["top"] == "edge":
            device_array[0:buffer_thickness, :] = 0
        if buffer_mode["bottom"] == "edge":
            device_array[-buffer_thickness:, :] = 0
        if buffer_mode["left"] == "edge":
            device_array[:, 0:buffer_thickness] = 0
        if buffer_mode["right"] == "edge":
            device_array[:, -buffer_thickness:] = 0

        ndarray = geometry.trim(device_array=device_array)
        return ndarray

    def to_img(self, img_path: str = "prefab_device.png"):
        cv2.imwrite(img_path, 255 * self.to_ndarray())
        print(f"Saved Device to '{img_path}'")

    def to_gds(
        self,
        gds_path: str = "prefab_device.gds",
        cell_name: str = "prefab_device",
        gds_layer: tuple[int, int] = (1, 0),
        contour_approx_mode: int = 2,
    ):
        gdstk_cell = self._device_to_gdstk(
            cell_name=cell_name,
            gds_layer=gds_layer,
            contour_approx_mode=contour_approx_mode,
        )
        gdstk_library = gdstk.Library()
        gdstk_library.add(gdstk_cell)
        gdstk_library.write_gds(outfile=gds_path, max_points=8190)
        print(f"Saved GDS to '{gds_path}'")

    def to_gdstk(
        self,
        cell_name: str = "prefab_device",
        gds_layer: tuple[int, int] = (1, 0),
        contour_approx_mode: int = 2,
    ):
        gdstk_cell = self._device_to_gdstk(
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
        self, show_buffer: bool = True, ax: Optional[Axes] = None, **kwargs
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_ylabel("y (nm)")
        ax.set_xlabel("x (nm)")

        if show_buffer:
            self._add_buffer_visualization(ax)
        return ax

    def plot(
        self, show_buffer: bool = True, ax: Optional[Axes] = None, **kwargs
    ) -> Axes:
        ax = self._plot_base(show_buffer=show_buffer, ax=ax, **kwargs)
        _ = ax.imshow(
            self.device_array,
            extent=[0, self.device_array.shape[1], 0, self.device_array.shape[0]],
            **kwargs,
        )
        return ax

    def plot_contour(
        self,
        linewidth: Optional[int] = None,
        label: Optional[str] = "Device contour",
        show_buffer: bool = True,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        ax = self._plot_base(show_buffer=show_buffer, ax=ax, **kwargs)
        kwargs.setdefault("cmap", "spring")
        if linewidth is None:
            linewidth = self.device_array.shape[0] // 100

        contours, _ = cv2.findContours(
            geometry.binarize_hard(self.device_array).astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_array = np.zeros_like(self.device_array, dtype=np.uint8)
        cv2.drawContours(contour_array, contours, -1, (255,), linewidth)
        contour_array = np.ma.masked_equal(contour_array, 0)

        _ = ax.imshow(
            contour_array,
            extent=[0, self.device_array.shape[1], 0, self.device_array.shape[0]],
            **kwargs,
        )

        cmap = cm.get_cmap(kwargs.get("cmap", "spring"))
        legend_proxy = Line2D([0], [0], linestyle="-", color=cmap(1))
        ax.legend([legend_proxy], [label], loc="upper right")
        return ax

    def plot_uncertainty(
        self, show_buffer: bool = True, ax: Optional[Axes] = None, **kwargs
    ):
        ax = self._plot_base(show_buffer=show_buffer, ax=ax, **kwargs)

        uncertainty_array = 1 - 2 * np.abs(0.5 - self.device_array)

        _ = ax.imshow(
            uncertainty_array,
            extent=[0, self.device_array.shape[1], 0, self.device_array.shape[0]],
            **kwargs,
        )

        cbar = plt.colorbar(_, ax=ax)
        cbar.set_label("Uncertainty (a.u.)")
        return ax

    def _add_buffer_visualization(self, ax):
        buffer_thickness = self.buffer_spec.thickness
        buffer_fill = (0, 1, 0, 0.2)
        buffer_hatch = "/"

        mid_rect = Rectangle(
            (buffer_thickness, buffer_thickness),
            self.device_array.shape[1] - 2 * buffer_thickness,
            self.device_array.shape[0] - 2 * buffer_thickness,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(mid_rect)

        top_rect = Rectangle(
            (0, 0),
            self.device_array.shape[1],
            buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(top_rect)

        bottom_rect = Rectangle(
            (0, self.device_array.shape[0] - buffer_thickness),
            self.device_array.shape[1],
            buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(bottom_rect)

        left_rect = Rectangle(
            (0, buffer_thickness),
            buffer_thickness,
            self.device_array.shape[0] - 2 * buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(left_rect)

        right_rect = Rectangle(
            (
                self.device_array.shape[1] - buffer_thickness,
                buffer_thickness,
            ),
            buffer_thickness,
            self.device_array.shape[0] - 2 * buffer_thickness,
            facecolor=buffer_fill,
            hatch=buffer_hatch,
        )
        ax.add_patch(right_rect)

    def normalize(self) -> "Device":
        normalized_device_array = geometry.normalize(device_array=self.device_array)
        return self.model_copy(update={"device_array": normalized_device_array})

    def binarize(self, eta: float = 0.5, beta: float = np.inf) -> "Device":
        binarized_device_array = geometry.binarize(
            device_array=self.device_array, eta=eta, beta=beta
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def binarize_hard(self, eta: float = 0.5) -> "Device":
        binarized_device_array = geometry.binarize_hard(
            device_array=self.device_array, eta=eta
        )
        return self.model_copy(update={"device_array": binarized_device_array})

    def ternarize(self, eta1: float = 1 / 3, eta2: float = 2 / 3) -> "Device":
        ternarized_device_array = geometry.ternarize(
            device_array=self.device_array, eta1=eta1, eta2=eta2
        )
        return self.model_copy(update={"device_array": ternarized_device_array})

    def trim(self) -> "Device":
        trimmed_device_array = geometry.trim(
            device_array=self.device_array,
            buffer_thickness=self.buffer_spec.thickness,
        )
        return self.model_copy(update={"device_array": trimmed_device_array})

    def blur(self, sigma: float = 1.0) -> "Device":
        blurred_device_array = geometry.blur(
            device_array=self.device_array, sigma=sigma
        )
        return self.model_copy(update={"device_array": blurred_device_array})
