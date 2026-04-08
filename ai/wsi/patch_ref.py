from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PatchRef:
    """
    Reference metadata for a single WSI patch.

    This object does not store the actual image patch.
    It only stores the information needed to load that patch later.

    Coordinate convention
    ---------------------
    OpenSlide.read_region(location, level, size) expects:
    - location: (x, y) in level-0 coordinates
    - level: pyramid level to read from
    - size: (width, height) in pixels at that level

    Therefore, in this PatchRef:
    - x, y are level-0 coordinates
    - width, height are pixel sizes at read_level
    """

    image_path: Path  #path to WSI file
    x: int            #top left x coordinate of patch (lv0)
    y: int            #top left y coordinate of patch (lv0)
    width: int 
    height: int
    read_level: int   #which lv to read from
    mpp_x: float
    mpp_y: float

    # Runs right after init to reject invalid patch references
    def __post_init__(self) -> None:
        if not isinstance(self.image_path, Path):
            raise TypeError(f"image_path must be a Path, got {type(self.image_path).__name__}")
        
        #Handling negative values of coordinates or level
        if self.x < 0: 
            raise ValueError(f"x must be >= 0, got {self.x}")
        if self.y < 0:
            raise ValueError(f"y must be >= 0, got {self.y}")
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if self.read_level < 0:
            raise ValueError(f"read_level must be >= 0, got {self.read_level}")
        
    # Formatting eg. ref.x -> x, or ref.height -> height
    @property
    def level0_pos(self) -> tuple[int, int]:
        """Top-left patch location in level-0 coordinates."""
        return (self.x, self.y)

    @property
    def read_size(self) -> tuple[int, int]:
        """Patch size in pixels at read_level."""
        return (self.width, self.height)

    @property
    def mpp(self) -> tuple[float, float]:
        """WSI microns-per-pixel metadata."""
        return (self.mpp_x, self.mpp_y)

    # Converts object into a plain Python dictionary for logging or debugging afterwards
    def to_dict(self) -> dict:
        """Convert to a plain dictionary for logging or JSON export."""
        return {
            "image_path": str(self.image_path),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "read_level": self.read_level,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
        }
    
    #Defines how the object should look when printed
    def __repr__(self) -> str:
        return (
            "PatchRef("
            f"image_path='{self.image_path}', "
            f"x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, "
            f"read_level={self.read_level}, "
            f"mpp_x={self.mpp_x}, mpp_y={self.mpp_y}"
            ")"
        )
