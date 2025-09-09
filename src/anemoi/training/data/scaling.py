# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

from typing import Any, Dict
import numpy as np

LOGGER = logging.getLogger(__name__)


class BasePressureLevelScaler(ABC):
    """Configurable method converting pressure level of variable to PTL scaling.

    Scaling variables depending on pressure levels (50 to 1000).
    """

    def __init__(self, slope: float = 1.0 / 1000, minimum: float = 0.0) -> None:
        """Initialise Scaler with slope and minimum.

        Parameters
        ----------
        slope : float
            Slope of the scaling function.
        minimum : float
            Minimum value of the scaling function.

        """
        self.slope = slope
        self.minimum = minimum

    @abstractmethod
    def scaler(self, plev: float) -> np.ndarray: ...


class StepReluPressureLevelScaler(BasePressureLevelScaler):
    """Relu pressure scaler with steps dependening on threshold levels
    """

    def __init__(self, steps: Dict[str,Any]) -> None:
        """Initialise Scaler with slope and minimum.

        Parameters
        ----------
        slope : float
            Slope of the scaling function.
        minimum : float
            Minimum value of the scaling function.

        """
        super().__init__(slope=0.001, minimum=0.0)
        self.steps = steps

        for step in steps:
            assert 'slope' in steps[step]
            assert 'minimum' in steps[step]
    
    def _get_step(self, level: int) -> int:
        level_step: int | None = None
        for step in self.steps:
            if level >= step and (level_step is None or step > level_step):
                level_step = step
        if level_step is None:
            raise ValueError(f"No valid step for level {level}.")
        return level_step
    
    def scaler(self, plev: float) -> np.ndarray:
        step_config = self.steps[self._get_step(plev)]        
        return max(float(step_config['minimum']), plev * float(step_config['slope']))

class LinearPressureLevelScaler(BasePressureLevelScaler):
    """Linear with slope self.slope, yaxis shift by self.minimum."""

    def scaler(self, plev: float) -> np.ndarray:
        return plev * self.slope + self.minimum


class ReluPressureLevelScaler(BasePressureLevelScaler):
    """Linear above self.minimum, taking constant value self.minimum below."""

    def scaler(self, plev: float) -> np.ndarray:
        return max(self.minimum, plev * self.slope)


class PolynomialPressureLevelScaler(BasePressureLevelScaler):
    """Polynomial scaling, (slope * plev)^2, yaxis shift by self.minimum."""

    def scaler(self, plev: float) -> np.ndarray:
        return (self.slope * plev) ** 2 + self.minimum


class NoPressureLevelScaler(BasePressureLevelScaler):
    """Constant scaling by 1.0."""

    def __init__(self, slope: float = 0.0, minimum: float = 1.0) -> None:
        """Initialise Scaler with constant scaling of 1."""
        assert (
            minimum == 1.0 and slope == 0
        ), "self.minimum must be 1.0 and self.slope 0.0 for no scaling to fit with definition of linear function."
        super().__init__(slope=0.0, minimum=1.0)

    @staticmethod
    def scaler(plev: float) -> np.ndarray:
        del plev  # unused
        # no scaling, always return 1.0
        return 1.0
