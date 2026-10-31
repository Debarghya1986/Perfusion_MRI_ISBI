import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod


def _activation_broadcast_view(values, x):
    """Broadcast one value per activation over an arbitrary tensor rank."""
    return values.reshape(1, -1, *([1] * (x.ndim - 2)))


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):

        # The value of the spline at any x is a combination
        # of at most two coefficients
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(
            min=x_min, max=x_max - step_size)

        floored_x = torch.floor((x_clamped - x_min) /
                                step_size)  # left coefficient

        # distance to left coefficient
        fracs = (x - x_min) / step_size - floored_x

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (_activation_broadcast_view(zero_knot_indexes, x)
                   + floored_x).long()

        coefficients_vect = coefficients.view(-1)

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        # ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients, indexes, step_size = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / step_size * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(
            coefficients_vect, dtype=coefficients_vect.dtype)
        # right coefficients gradients

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None


class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):

        # The value of the spline at any x is a combination
        # of at most two coefficients
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(
            min=x_min, max=x_max - step_size)

        floored_x = torch.floor((x_clamped - x_min) /
                                step_size)  # left coefficient

        # distance to left coefficient
        fracs = (x - x_min) / step_size - floored_x

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (_activation_broadcast_view(zero_knot_indexes, x)
                   + floored_x).long()

        coefficients_vect = coefficients.view(-1)
        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = (
            coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / step_size

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients, indexes, step_size = ctx.saved_tensors
        grad_x = 0 * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients.view(-1))
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            torch.ones_like(fracs).view(-1) / step_size)
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            -torch.ones_like(fracs).view(-1) / step_size)

        return grad_x, grad_coefficients_vect, None, None, None, None


class Quadratic_Spline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(
            min=x_min, max=x_max - 2*step_size)

        floored_x = torch.floor((x_clamped - x_min) / step_size)
        floored_x = floored_x.clamp(min=0, max=num_knots - 3)

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (_activation_broadcast_view(zero_knot_indexes, x)
                   + floored_x).long()
        max_idx = coefficients.numel() - 3  # need indexes, indexes+1, indexes+2
        indexes = indexes.clamp(min=0, max=max_idx)

        # ------------------------------------------------------------------
        # Interior B-spline evaluation on [x_min, x_max - 2*step_size].
        # Uses CLAMPED x so that the formulas stay well-defined; out-of-range
        # points get their output overridden by the affine extension below,
        # so what the interior formula computes for them is discarded anyway.
        # ------------------------------------------------------------------
        shift1 = (x_clamped - x_min) / step_size - floored_x

        frac1 = ((shift1 - 1)**2)/2
        frac2 = (-2*(shift1)**2 + 2*shift1 + 1)/2
        frac3 = (shift1)**2/2

        coefficients_vect = coefficients.view(-1)

        # ------------------------------------------------------------------
        # C^1 affine extensions outside [x_min, x_max - 2*step_size].
        #
        # At the left boundary  (shift=0, indexes clamped to 0):
        #     Q(x_min) = (c_0 + c_1)/2,   Q'(x_min) = (c_1 - c_0)/step
        # At the right boundary (shift=0, indexes clamped to N-3):
        #     Q(x_rb)  = (c_{N-3} + c_{N-2})/2,
        #     Q'(x_rb) = (c_{N-2} - c_{N-3})/step
        #
        # Integrating the constant-extended linear spline (slope frozen to
        # the boundary value) gives an affine Q on each tail, with both
        # value AND slope continuous at the boundary -> globally C^1.
        #
        # Written in basis form against the same (c_idx, c_{idx+1}, c_{idx+2})
        # triple the interior formula uses, both tails collapse to the SAME
        # frac pattern — c_{idx+2} is unused at both boundaries in this
        # file's convention, so a single override handles both sides:
        #
        #     frac1 = 1/2 - shift_raw   (weight on c_idx)
        #     frac2 = 1/2 + shift_raw   (weight on c_{idx+1})
        #     frac3 = 0                 (weight on c_{idx+2})
        #
        # where shift_raw = (x - x_min)/step_size - floored_x uses the
        # UN-clamped x (so it is < 0 on the left tail and > 0 on the right).
        # The backward scatter_add is unchanged — with the overridden fracs
        # it routes gradients to the correct boundary coefficients for free.
        # ------------------------------------------------------------------
        out_of_range = (x < x_min) | (x > x_max - 2*step_size)

        if out_of_range.any():
            shift_raw = (x - x_min) / step_size - floored_x
            half = torch.tensor(0.5, dtype=shift_raw.dtype,
                                device=shift_raw.device)
            frac1 = torch.where(out_of_range, half - shift_raw, frac1)
            frac2 = torch.where(out_of_range, half + shift_raw, frac2)
            frac3 = torch.where(
                out_of_range, torch.zeros_like(shift_raw), frac3)

        activation_output = coefficients_vect[indexes + 2] * frac3 + \
            coefficients_vect[indexes + 1] * frac2 + \
            coefficients_vect[indexes] * frac1

        grad_x = coefficients_vect[indexes + 2] * (shift1) + \
            coefficients_vect[indexes + 1] * (1 - 2*shift1) + \
            coefficients_vect[indexes] * ((shift1 - 1))

        grad_x = grad_x / step_size

        # Override x-gradient on the affine tails with the constant boundary
        # slope (c_{idx+1} - c_idx)/step, with idx clamped to 0 on the left
        # and N-3 on the right — exactly the interior derivative at shift=0
        # at those clamped intervals, giving C^1 continuity.
        if out_of_range.any():
            boundary_slope = (coefficients_vect[indexes + 1]
                              - coefficients_vect[indexes]) / step_size
            grad_x = torch.where(out_of_range, boundary_slope, grad_x)

        ctx.save_for_backward(grad_x, frac1, frac2, frac3,
                              coefficients, indexes, step_size)

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        grad_x, frac1, frac2, frac3, coefficients, indexes, grid = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = grad_x * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 2,
                                            (frac3 * grad_out).view(-1))

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (frac2 * grad_out).view(-1))

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1),
                                            (frac1 * grad_out).view(-1))

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None
