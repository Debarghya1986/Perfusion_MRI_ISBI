import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod
import core.activations.spline_autograd_func as spline_autograd_func

import time


class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_knots (int): number of knots of the spline
        num_activations (int) : number of activation functions
        x_min (float): position of left-most knot
        x_max (float): position of right-most knot
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
        antisymmetric (bool): Constrain the potential to be symmetric <=> activation antisymmetric
    """

    def __init__(self, num_activations, num_knots, x_min, x_max, init, slope_max=None, slope_min=None, clamp=True, **kwargs):

        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        # self.x_min = torch.tensor([x_min])
        # self.x_max = torch.tensor([x_max])
        self.slope_min = slope_min
        self.slope_max = slope_max

        # self.step_size = (self.x_max - self.x_min) / (self.num_knots - 1)
        
        # Non-persistent buffers: auto-track device, excluded from state_dict
        self.register_buffer("x_min", torch.tensor([x_min]), persistent=False)
        self.register_buffer("x_max", torch.tensor([x_max]), persistent=False)
        self.register_buffer(
            "step_size",
            (self.x_max - self.x_min) / (self.num_knots - 1),
            persistent=False,
        )
        
        self.clamp = clamp
        self.no_constraints = (
            slope_max is None and slope_min is None and not clamp)

        # parameters
        coefficients = self.initialize_coeffs()  # spline coefficients
        self.coefficients = nn.Parameter(coefficients)

        self.projected_coefficients_cached = None
        self.register_buffer("D2_filter", Tensor(
            [1, -2, 1]).view(1, 1, 3).div(self.step_size), persistent=False)

        self.init_zero_knot_indexes()

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        # self.zero_knot_indexes = (activation_arange * self.num_knots)
        self.register_buffer(
            "zero_knot_indexes",
            activation_arange * self.num_knots,
            persistent=False,
        )

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        init = self.init
        grid_tensor = torch.linspace(self.x_min.squeeze(), self.x_max.squeeze(), self.num_knots).expand((self.num_activations, self.num_knots)).clone()

        if isinstance(init, float):
            coefficients = torch.ones_like(grid_tensor)*init
        elif init == 'gaussian':
            coefficients = torch.exp(-grid_tensor**2)
        elif init == 'identity':
            coefficients = grid_tensor
        elif init == 'zero':
            coefficients = grid_tensor*0
        else:
            raise ValueError('init should be in [identity, zero].')

        return coefficients

    @property
    def projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        else:
            return (self.clipped_coefficients())

    def cached_projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self.clipped_coefficients()

    @property
    def slopes(self):
        """ Get the slopes of the activations
        """
        coeff = self.projected_coefficients
        slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.step_size

        return slopes

    @property
    def device(self):
        return self.coefficients.device

    # ------------------------------------------------------------------
    # Compatibility helpers used by the uploaded CCD3D model
    # ------------------------------------------------------------------
    @property
    def grid(self):
        """Alias for the knot spacing expected by the CCD3D contractivity check."""
        return self.step_size

    @property
    def monotonic_constraint(self):
        """True when the configured slope interval enforces a monotone spline."""
        return self.slope_min is not None and float(self.slope_min) >= 0.0

    @property
    def slope_min_attr(self):
        """Legacy increment-form lower bound expected by CCD3D."""
        if self.slope_min is None:
            return None
        return float(self.slope_min) * float(self.step_size.item())

    @property
    def slope_max_attr(self):
        """Legacy increment-form upper bound expected by CCD3D."""
        if self.slope_max is None:
            return None
        return float(self.slope_max) * float(self.step_size.item())

    def integrate(self, x):
        """Analytic antiderivative of the projected linear spline, anchored at 0.

        The output has the same shape as ``x`` and satisfies d/dx integrate(x)
        = forward(x) almost everywhere.  This is used only by ``CCD3D.potential``;
        denoiser training and PnP call ``forward`` directly.
        """
        in_shape = x.shape
        in_channels = int(in_shape[1])
        if in_channels % self.num_activations != 0:
            raise ValueError(
                'Number of input channels must be divisible by number of activations.')

        z = x.reshape(
            x.shape[0], self.num_activations,
            in_channels // self.num_activations, *x.shape[2:]
        )
        coeff = self.projected_coefficients
        step = self.step_size.to(device=x.device, dtype=x.dtype)
        xmin = self.x_min.to(device=x.device, dtype=x.dtype)
        xmax = self.x_max.to(device=x.device, dtype=x.dtype)
        n = self.num_knots

        slopes = (coeff[:, 1:] - coeff[:, :-1]) / step
        # cumulative trapezoid area from xmin to each knot
        seg_area = 0.5 * (coeff[:, :-1] + coeff[:, 1:]) * step
        cum = torch.cat(
            [torch.zeros(coeff.shape[0], 1, device=x.device, dtype=x.dtype),
             torch.cumsum(seg_area, dim=1)],
            dim=1,
        )

        def primitive_from_xmin(v):
            # segment index for interior; tails handled below
            idx = torch.floor((v - xmin) / step).long().clamp(0, n - 2)
            # broadcast activation index to v rank
            aidx = torch.arange(self.num_activations, device=x.device)
            aidx = aidx.reshape(1, -1, *([1] * (v.ndim - 2))).expand_as(idx)
            c0 = coeff[aidx, idx]
            sl = slopes[aidx, idx]
            base = cum[aidx, idx]
            xk = xmin + idx.to(v.dtype) * step
            d = v - xk
            interior = base + c0 * d + 0.5 * sl * d.square()

            left_d = v - xmin
            left = coeff[:, 0].reshape(1, -1, *([1] * (v.ndim - 2))) * left_d \
                + 0.5 * slopes[:, 0].reshape(1, -1, *([1] * (v.ndim - 2))) * left_d.square()

            total = cum[:, -1].reshape(1, -1, *([1] * (v.ndim - 2)))
            right_d = v - xmax
            right = total \
                + coeff[:, -1].reshape(1, -1, *([1] * (v.ndim - 2))) * right_d \
                + 0.5 * slopes[:, -1].reshape(1, -1, *([1] * (v.ndim - 2))) * right_d.square()

            return torch.where(v < xmin, left, torch.where(v > xmax, right, interior))

        value = primitive_from_xmin(z)
        zero = torch.zeros_like(z)
        anchor = primitive_from_xmin(zero)
        return (value - anchor).reshape(in_shape)

    def hyper_param_to_device(self):
        device = self.device
        self.x_min, self.x_max, self.step_size, self.zero_knot_indexes = self.x_min.to(
            device), self.x_max.to(device), self.step_size.to(device), self.zero_knot_indexes.to(device)

    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                A tensor whose channel dimension is dimension 1.  Both 4D
                images and 5D spatiotemporal volumes are supported.

        Returns:
            output (torch.Tensor)
        """
        # self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = int(in_shape[1])

        if in_channels % self.num_activations != 0:
            raise ValueError(
                'Number of input channels must be divisible by number of activations.')

        x = x.reshape(x.shape[0], self.num_activations,
                      in_channels // self.num_activations, *x.shape[2:])

        x = spline_autograd_func.LinearSpline_Func.apply(
            x, self.projected_coefficients, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)

        x = x.reshape(in_shape)

        return x

    def extra_repr(self):
        """repr that is compatible with x_min/x_max being registered buffers."""
        return (
            f"num_activations={self.num_activations}, "
            f"init={self.init}, num_knots={self.num_knots}, "
            f"range=[{self.x_min.item():.3f}, {self.x_max.item():.3f}], "
            f"slope_max={self.slope_max}, slope_min={self.slope_min}."
        )

    def clipped_coefficients(self):
        """Simple projection of the spline coefficients to enforce the constraints, for e.g. bounded slope"""

        device = self.device

        if self.no_constraints:
            return (self.coefficients)

        cs = self.coefficients

        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.step_size

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(
                new_slopes, self.slope_min, self.slope_max)

        # clamp extension
        if self.clamp:
            new_slopes[:, 0] = 0
            new_slopes[:, -1] = 0

        new_cs = torch.zeros(self.coefficients.shape,
                             device=device, dtype=cs.dtype)

        new_cs[:, 1:] = torch.cumsum(new_slopes, dim=1) * self.step_size

        new_cs = new_cs + (cs - new_cs).mean(dim=1).unsqueeze(1)

        return new_cs

    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        return F.conv1d(self.projected_coefficients.unsqueeze(1), self.D2_filter).squeeze(1)

    def tv2(self):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1).sum()


class QuadraticSpline(ABC, nn.Module):
    """
    Quadratic B-spline activation function.
 
    The spline is evaluated on [x_min, x_max - 2*step] using Quadratic_Spline_Func.
    Outside that range:
 
        clamp=True  -> affine (C1) extensions at both ends:
                       f_ext(x) = f(boundary) + f'(boundary) * (x - boundary)
        clamp=False -> no extensions / no constraints (raw quadratic evaluation)
 
    Constant extensions are intentionally avoided because a slope discontinuity
    at the boundary breaks differentiability of the composed map.
 
    Args:
        num_activations (int): number of independent activation functions
        num_knots (int):       number of B-spline knots
        x_min (float):         left-most knot position
        x_max (float):         right-most knot position
        init (str or float):   coefficient initialization
        slope_max (float|None): upper bound on inter-knot slopes
        slope_min (float|None): lower bound on inter-knot slopes
        clamp (bool):          True = affine extensions, False = unconstrained
    """

    def __init__(self, num_activations, num_knots, x_min, x_max, init,
                 slope_max=None, slope_min=None, clamp=True, **kwargs):
        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.clamp = clamp
        self.no_constraints = (
            slope_max is None and slope_min is None and not clamp)

        # Non-persistent buffers: auto-track device, excluded from state_dict
        self.register_buffer("x_min", torch.tensor([x_min]), persistent=False)
        self.register_buffer("x_max", torch.tensor([x_max]), persistent=False)
        self.register_buffer(
            "step_size",
            (self.x_max - self.x_min) / (self.num_knots - 1),
            persistent=False,
        )

        # Learnable spline coefficients
        coefficients = self._initialize_coeffs()
        self.coefficients = nn.Parameter(coefficients)

        self.projected_coefficients_cached = None

        # Second-order finite difference filter for TV(2) regularization
        self.register_buffer(
            "D2_filter",
            Tensor([1, -2, 1]).view(1, 1, 3).div(self.step_size), persistent=False
        )

        self._init_zero_knot_indexes()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_zero_knot_indexes(self):
        activation_arange = torch.arange(0, self.num_activations)
        self.register_buffer(
            "zero_knot_indexes",
            activation_arange * self.num_knots,
            persistent=False,
        )

    def _initialize_coeffs(self):
        grid = torch.linspace(
            self.x_min.squeeze(), self.x_max.squeeze(), self.num_knots,
        ).expand(self.num_activations, self.num_knots).clone()

        if isinstance(self.init, float):
            return torch.ones_like(grid) * self.init
        elif self.init == 'gaussian':
            return torch.exp(-grid ** 2)
        elif self.init == 'identity':
            return grid.clone()
        elif self.init == 'zero':
            return torch.zeros_like(grid)
        else:
            raise ValueError(
                f"init must be in [identity, zero, gaussian] or a float, got '{self.init}'")

    # ------------------------------------------------------------------
    # Coefficient projection / constraints
    # ------------------------------------------------------------------

    @property
    def projected_coefficients(self):
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        return self._clipped_coefficients()

    def cached_projected_coefficients(self):
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self._clipped_coefficients()

    def _clipped_coefficients(self):
        """
        Project coefficients to satisfy slope bounds only.
 
        For a quadratic B-spline the derivative at the left edge of segment i
        (shift1=0) is  (c[i+1] - c[i]) / step.
 
        Boundary slopes are left free — when clamp=True the affine extension
        in forward() uses them directly. No slope is forced to zero.
        """
        if self.no_constraints:
            return self.coefficients

        cs = self.coefficients
        step = self.step_size
        device = cs.device

        # Inter-knot slopes: (num_activations, num_knots-1)
        new_slopes = (cs[:, 1:] - cs[:, :-1]) / step

        # Global slope bounds
        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(
                new_slopes, self.slope_min, self.slope_max)

        # Reconstruct coefficients from projected slopes
        new_cs = torch.zeros_like(cs)
        new_cs[:, 1:] = torch.cumsum(new_slopes, dim=1) * step
        # Shift so the mean matches the original (preserves DC level)
        new_cs = new_cs + (cs - new_cs).mean(dim=1, keepdim=True)

        return new_cs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def slopes(self):
        coeff = self.projected_coefficients
        return (coeff[:, 1:] - coeff[:, :-1]) / self.step_size

    @property
    def device(self):
        return self.coefficients.device

    @property
    def relu_slopes(self):
        return F.conv1d(
            self.projected_coefficients.unsqueeze(1), self.D2_filter,
        ).squeeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _boundary_slopes(self, coeff):
        """
        Compute boundary derivatives from the projected coefficients.
 
        Left  (segment 0, shift1=0):         f'(x_min)   = (c[1] - c[0])     / step
        Right (last segment, shift1=0):       f'(x_right) = (c[-2] - c[-3])   / step
 
        Returns:
            left_slopes:  (num_activations,)
            right_slopes: (num_activations,)
        """
        step = self.step_size
        left_slopes = (coeff[:, 1] - coeff[:, 0]) / step
        right_slopes = (coeff[:, -2] - coeff[:, -3]) / step
        return left_slopes, right_slopes

    def forward(self, x):
        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError(
                'Number of input channels must be divisible by number of activations.')

        x = x.reshape(
            x.shape[0], self.num_activations,
            in_channels // self.num_activations, *x.shape[2:],
        )

        coeff = self.projected_coefficients

        # Core quadratic B-spline evaluation (clamps x internally)
        output = spline_autograd_func.Quadratic_Spline_Func.apply(
            x, coeff, self.x_min, self.x_max,
            self.num_knots, self.zero_knot_indexes,
        )

        # ----- Affine extensions outside the evaluation range -----
        # The autograd func clamps x to [x_min, x_max - 2*step], which
        # gives a constant extension by default.  When clamp=True we
        # replace that with a C1-continuous affine (linear) tail using
        # the boundary derivative of the quadratic spline.
        # When clamp=False we leave the raw (clamped) evaluation as-is.
        if self.clamp:
            left_slopes, right_slopes = self._boundary_slopes(coeff)

            # The spline wrapper inserts one sub-channel dimension, so a 5D
            # Conv3d tensor is 6D here.  Build the broadcast shape from the
            # actual rank instead of hard-coding the image case.
            broadcast_shape = (1, -1, *([1] * (x.ndim - 2)))
            left_slopes = left_slopes.reshape(broadcast_shape)
            right_slopes = right_slopes.reshape(broadcast_shape)

            x_right = self.x_max - 2 * self.step_size

            # Left tail:  x < x_min  =>  output += f'(x_min) * (x - x_min)
            left_mask = (x < self.x_min)
            if left_mask.any():
                output = output + left_slopes * (x - self.x_min) * left_mask

            # Right tail: x > x_right  =>  output += f'(x_right) * (x - x_right)
            right_mask = (x > x_right)
            if right_mask.any():
                output = output + right_slopes * (x - x_right) * right_mask

        output = output.reshape(in_shape)
        return output

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def tv2(self):
        """Second-order total-variation: TV(2) = ||relu_slopes||_1."""
        return self.relu_slopes.norm(1, dim=1).sum()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self):
        return (
            f'num_activations={self.num_activations}, '
            f'init={self.init}, num_knots={self.num_knots}, '
            f'range=[{self.x_min.item():.3f}, {self.x_max.item():.3f}], '
            f'slope_max={self.slope_max}, slope_min={self.slope_min}, '
            f'clamp={self.clamp}'
        )
