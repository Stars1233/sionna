#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for Turbo code encoding."""

import math
from typing import Optional, Tuple, Union

import torch

from sionna.phy import Block
from sionna.phy.fec import interleaving
from sionna.phy.fec.conv.encoding import ConvEncoder
from sionna.phy.fec.conv.utils import Trellis
from sionna.phy.fec.turbo.utils import (
    polynomial_selector,
    puncture_pattern,
    TurboTermination,
)

__all__ = ["TurboEncoder"]


class TurboEncoder(Block):
    r"""Performs encoding of information bits to a Turbo code codeword.

    Implements the standard Turbo code framework :cite:p:`Berrou`: Two identical
    rate-1/2 convolutional encoders
    :class:`~sionna.phy.fec.conv.encoding.ConvEncoder` are combined to produce
    a rate-1/3 Turbo code. Further, puncturing to attain a rate-1/2 Turbo code
    is supported.

    :param gen_poly: Tuple of strings with each string being a 0,1 sequence.
        If `None`, ``constraint_length`` must be provided.
    :param constraint_length: Valid values are between 3 and 6 inclusive.
        Only required if ``gen_poly`` is `None`.
    :param rate: Valid values are 1/3 and 1/2. Note that ``rate`` here denotes
        the *design* rate of the Turbo code. If ``terminate`` is `True`, a
        small rate-loss occurs.
    :param terminate: Underlying convolutional encoders are terminated to all
        zero state if `True`. If terminated, the true rate of the code is
        slightly lower than ``rate``.
    :param interleaver_type: Determines the choice of the interleaver to
        interleave the message bits before input to the second convolutional
        encoder. Valid values are `"3GPP"` or `"random"`. If `"3GPP"`, the
        Turbo code interleaver from the 3GPP LTE standard
        :cite:p:`3GPPTS36212` is used. If `"random"`, a random interleaver
        is used.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [..., k], `torch.float`.
        Tensor of information bits where `k` is the information length.

    :output cw: [..., k/rate], `torch.float`.
        Tensor where `rate` is provided as input parameter. The output is the
        encoded codeword for the input information tensor. When ``terminate``
        is `True`, the effective rate of the Turbo code is slightly less than
        ``rate``.

    .. rubric:: Notes

    Various notations are used in literature to represent the generator
    polynomials for convolutional codes. For simplicity
    :class:`~sionna.phy.fec.turbo.encoding.TurboEncoder` only
    accepts the binary format, i.e., `10011`, for the ``gen_poly`` argument
    which corresponds to the polynomial :math:`1 + D^3 + D^4`.

    Note that Turbo codes require the underlying convolutional encoders
    to be recursive systematic encoders. Only then the channel output
    from the systematic part of the first encoder can be used to decode
    the second encoder.

    Also note that ``constraint_length`` and ``memory`` are two different
    terms often used to denote the strength of the convolutional code. In
    this sub-package we use ``constraint_length``. For example, the polynomial
    `10011` has a ``constraint_length`` of 5, however its ``memory`` is
    only 4.

    When ``terminate`` is `True`, the true rate of the Turbo code is
    slightly lower than ``rate``. It can be computed as
    :math:`\frac{k}{\frac{k}{r}+\frac{4\mu}{3r}}` where `r` denotes
    ``rate`` and :math:`\mu` is the ``constraint_length`` - 1. For example,
    in 3GPP, ``constraint_length`` = 4, ``terminate`` = `True`, for
    ``rate`` = 1/3, true rate is equal to :math:`\frac{k}{3k+12}`.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.turbo import TurboEncoder

        encoder = TurboEncoder(rate=1/3, constraint_length=4, terminate=True)
        u = torch.randint(0, 2, (10, 40), dtype=torch.float32)
        c = encoder(u)
        print(c.shape)
        # torch.Size([10, 132])
    """

    def __init__(
        self,
        gen_poly: Optional[Tuple[str, ...]] = None,
        constraint_length: int = 3,
        rate: float = 1 / 3,
        terminate: bool = False,
        interleaver_type: str = "3GPP",
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if gen_poly is not None:
            if not all(isinstance(poly, str) for poly in gen_poly):
                raise TypeError("Each element of gen_poly must be a string.")
            if not all(len(poly) == len(gen_poly[0]) for poly in gen_poly):
                raise ValueError("Each polynomial must be of same length.")
            if not all(
                all(char in ["0", "1"] for char in poly) for poly in gen_poly
            ):
                raise ValueError("Each Polynomial must be a string of 0/1 s.")
            if len(gen_poly) != 2:
                raise ValueError(
                    "Generator polynomials need to be of rate-1/2."
                )
            self._gen_poly = gen_poly
        else:
            valid_constraint_length = (3, 4, 5, 6)
            if constraint_length not in valid_constraint_length:
                raise ValueError("Constraint length must be between 3 and 6.")
            self._gen_poly = polynomial_selector(constraint_length)

        valid_rates = (1 / 2, 1 / 3)
        if rate not in valid_rates:
            raise ValueError("Invalid coderate.")
        if not isinstance(terminate, bool):
            raise TypeError("terminate must be bool.")
        if interleaver_type not in ("3GPP", "random"):
            raise ValueError("Invalid interleaver_type.")

        self._coderate_desired = rate
        self._coderate = self._coderate_desired
        self._terminate = terminate
        self._interleaver_type = interleaver_type

        # Underlying convolutional encoders to be RSC
        rsc = True

        self._coderate_conv = 1 / len(self.gen_poly)
        self._punct_pattern = puncture_pattern(
            rate, self._coderate_conv, device=self.device
        )

        self._trellis = Trellis(self.gen_poly, rsc=rsc, device=self.device)
        self._mu = self._trellis._mu

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        # For conv codes, the code dimensions are unknown during initialization
        self._k: Optional[int] = None
        self._n: Optional[int] = None
        self._num_syms: Optional[int] = None

        if self.terminate:
            self._turbo_term = TurboTermination(
                self._mu + 1, conv_n=self._conv_n, device=self.device
            )
        else:
            self._turbo_term = None

        if self._interleaver_type == "3GPP":
            self._internal_interleaver = interleaving.Turbo3GPPInterleaver(
                precision=precision, device=device
            )
        else:
            self._internal_interleaver = interleaving.RandomInterleaver(
                keep_batch_constant=True,
                keep_state=True,
                axis=-1,
                precision=precision,
                device=device,
            )

        # Precompute puncture indices
        if self._punct_pattern is not None:
            self.register_buffer("_punct_idx", torch.where(self._punct_pattern.flatten())[0])
        else:
            self.register_buffer("_punct_idx", None)

        self._convencoder = ConvEncoder(
            gen_poly=self._gen_poly,
            rsc=rsc,
            terminate=self._terminate,
            precision=precision,
            device=device,
        )

    @property
    def gen_poly(self) -> Tuple[str, ...]:
        """Generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def constraint_length(self) -> int:
        """Constraint length of the encoder."""
        return self._mu + 1

    @property
    def coderate(self) -> float:
        """Rate of the code used in the encoder."""
        if self.terminate and self._k is None:
            print(
                "Note that, due to termination, the true coderate is lower "
                "than the returned design rate. "
                "The exact true rate is dependent on the value of k and "
                "hence cannot be computed before the first call()."
            )
        elif self.terminate and self._k is not None:
            term_factor = 1 + math.ceil(4 * self._mu / 3) / self._k
            self._coderate = self._coderate_desired / term_factor
        return self._coderate

    @property
    def trellis(self) -> Trellis:
        """Trellis object used during encoding."""
        return self._trellis

    @property
    def terminate(self) -> bool:
        """Indicates if the convolutional encoders are terminated."""
        return self._terminate

    @property
    def punct_pattern(self) -> Optional[torch.Tensor]:
        """Puncturing pattern for the Turbo codeword."""
        return self._punct_pattern

    @property
    def k(self) -> Optional[int]:
        """Number of information bits per codeword."""
        if self._k is None:
            print(
                "Note: The value of k cannot be computed before the first "
                "call()."
            )
        return self._k

    @property
    def n(self) -> Optional[int]:
        """Number of codeword bits."""
        if self._n is None:
            print(
                "Note: The value of n cannot be computed before the first "
                "call()."
            )
        return self._n

    @property
    def internal_interleaver(
        self,
    ) -> Union[
        interleaving.Turbo3GPPInterleaver, interleaving.RandomInterleaver
    ]:
        """Internal interleaver used for the second encoder."""
        return self._internal_interleaver

    def _puncture_cw(
        self, cw: torch.Tensor, input_device: torch.device
    ) -> torch.Tensor:
        """Punctures the codeword using the puncturing pattern.

        :param cw: Codeword tensor of shape (bs, n, 3).
        :param input_device: Device for tensor operations.

        :output cw: Punctured codeword tensor.
        """
        # cw shape: (bs, n, 3) - transpose to (n, 3, bs)
        cw = cw.permute(1, 2, 0)
        cw_n = cw.shape[0]

        punct_pattern = self._punct_pattern
        punct_idx = self._punct_idx
        if punct_pattern.device != input_device:
            punct_pattern = punct_pattern.to(input_device)
            punct_idx = punct_idx.to(input_device)

        punct_period = punct_pattern.shape[0]
        mask_reps = cw_n // punct_period

        idx = punct_idx.repeat(mask_reps)
        idx_per_period = punct_idx.shape[0]
        idx_per_time = idx_per_period / punct_period

        # When tiling punct_pattern doesn't cover cw, delta_times > 0
        delta_times = cw_n - (mask_reps * punct_period)
        delta_idx_rows = int(delta_times * idx_per_time)

        # Create row offsets
        time_offset = punct_period * torch.arange(
            mask_reps, device=input_device
        )
        row_idx = time_offset.unsqueeze(0).expand(idx_per_period, -1)
        row_idx = row_idx.t().reshape(-1)

        total_indices = mask_reps * idx_per_period + delta_idx_rows

        if delta_times > 0:
            idx = torch.cat([idx, punct_idx[:delta_idx_rows]], dim=0)
            # Additional index row offsets
            time_n = punct_period * mask_reps
            row_idx_delta = torch.arange(
                time_n, time_n + delta_times, device=input_device
            )
            row_idx_delta = row_idx_delta.unsqueeze(0).expand(
                delta_idx_rows, -1
            )
            row_idx = torch.cat([row_idx, row_idx_delta.flatten()], dim=0)

        # Compute 2D indices
        # idx contains the flat position in the (punct_period, 3) pattern
        # We need to convert to (row, col) format
        row_in_pattern = idx // 3
        col_in_pattern = idx % 3

        # Add row offset to row_in_pattern
        final_row = row_in_pattern + row_idx[:total_indices]
        final_col = col_in_pattern

        # Gather the elements
        cw_flat = cw.reshape(-1, cw.shape[-1])  # (n*3, bs)
        flat_idx = final_row * 3 + final_col
        cw_punct = cw_flat[flat_idx]  # (num_punctured, bs)

        cw = cw_punct.t()  # (bs, num_punctured)
        return cw

    def build(self, input_shape: tuple) -> None:
        """Build block and check dimensions.

        :param input_shape: Shape of input tensor (..., k).
        """
        self._k = input_shape[-1]
        self._n = int(self._k / self._coderate_desired)

        if self._interleaver_type == "3GPP":
            if self._k > 6144:
                raise ValueError(
                    "3GPP Turbo Codes define Interleavers only "
                    "up to frame lengths of 6144."
                )

        # Num. of encoding periods/state transitions
        self._num_syms = int(self._k // self._conv_k)

    @torch.compiler.disable
    def call(self, bits: torch.Tensor, /) -> torch.Tensor:
        """Turbo code encoding function.

        :param bits: Information tensor of shape [..., k].

        :output cw: Encoded codeword tensor of shape [..., n].

        .. rubric:: Notes

        This method uses ``@torch.compiler.disable`` because the internal
        ConvEncoder calls cause slow compilation with ``torch.compile``.
        """
        # Use input device for all computations
        input_device = bits.device

        # Rebuild if k has changed
        if bits.shape[-1] != self._k:
            self._built = False
            self.build(bits.shape)
            self._built = True

        if self._terminate:
            num_term_bits_ = int(
                self._turbo_term.get_num_term_syms() / self._coderate_conv
            )
            num_term_bits_punct = int(
                num_term_bits_ * self._coderate_conv / self._coderate_desired
            )
        else:
            num_term_bits_ = 0
            num_term_bits_punct = 0

        output_shape = list(bits.shape)
        output_shape[-1] = self._n + num_term_bits_punct

        preterm_n = int(self._k / self._coderate_conv)
        msg = bits.to(torch.int32).reshape(-1, self._k)
        msg2 = self._internal_interleaver(msg)

        cw1_ = self._convencoder(msg)
        cw2_ = self._convencoder(msg2)

        cw1, term1 = cw1_[:, :preterm_n], cw1_[:, preterm_n:]
        cw2, term2 = cw2_[:, :preterm_n], cw2_[:, preterm_n:]

        # Gather parity stream from 2nd encoder
        par_idx = torch.arange(
            1, preterm_n, self._conv_n, device=input_device
        )
        cw2_par = cw2[:, par_idx]

        cw1 = cw1.reshape(-1, self._k, self._conv_n)
        cw2_par = cw2_par.reshape(-1, self._k, 1)

        # Concatenate 2nd encoder parity to _conv_n streams from first encoder
        cw = torch.cat([cw1, cw2_par], dim=-1)

        if self.terminate:
            term_syms_turbo = self._turbo_term.termbits_conv2turbo(term1, term2)
            term_syms_turbo = term_syms_turbo.reshape(-1, num_term_bits_ // 2, 3)
            cw = torch.cat([cw, term_syms_turbo], dim=-2)

        if self._punct_pattern is not None:
            cw = self._puncture_cw(cw, input_device)
        else:
            cw = cw.reshape(-1, cw.shape[1] * cw.shape[2])

        cw = cw.to(self.dtype)
        cw_reshaped = cw.reshape(output_shape)
        # Ensure contiguous output for torch.compile compatibility
        return cw_reshaped.contiguous()

