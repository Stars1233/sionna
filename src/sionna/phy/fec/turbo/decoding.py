#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Turbo code decoding."""

from typing import Optional, Union

import torch

from sionna.phy import Block
from sionna.phy.fec import interleaving
from sionna.phy.fec.conv.decoding import BCJRDecoder
from sionna.phy.fec.conv.utils import Trellis
from sionna.phy.fec.turbo.utils import (
    TurboTermination,
    polynomial_selector,
    puncture_pattern,
)

__all__ = ["TurboDecoder"]


class TurboDecoder(Block):
    r"""Turbo code decoder based on BCJR component decoders :cite:p:`Berrou`.

    Takes as input LLRs and returns LLRs or hard decided bits, i.e., an
    estimate of the information tensor.

    This decoder is based on the
    :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder` and, thus, internally
    instantiates two :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder` blocks.

    :param encoder: If ``encoder`` is provided as input, the following input
        parameters are not required and will be ignored: ``gen_poly``,
        ``rate``, ``constraint_length``, ``terminate``, ``interleaver``. They
        will be inferred from the ``encoder`` object itself.
        If ``encoder`` is `None`, the above parameters must be provided
        explicitly.
    :param gen_poly: Tuple of strings with each string being a 0, 1 sequence.
        If `None`, ``rate`` and ``constraint_length`` must be provided.
    :param rate: Rate of the Turbo code. Valid values are 1/3 and 1/2. Note
        that ``gen_poly``, if provided, is used to encode the underlying
        convolutional code, which traditionally has rate 1/2.
    :param constraint_length: Valid values are between 3 and 6 inclusive.
        Only required if ``encoder`` and ``gen_poly`` are `None`.
    :param interleaver: `"3GPP"` or `"random"`. If `"3GPP"`, the internal
        interleaver for Turbo codes as specified in :cite:p:`3GPPTS36212`
        will be used. Only required if ``encoder`` is `None`.
    :param terminate: If `True`, the two underlying convolutional encoders
        are assumed to have terminated to all zero state.
    :param num_iter: Number of iterations for the Turbo decoding to run.
        Each iteration of Turbo decoding entails one BCJR decoder for each
        of the underlying convolutional code components.
    :param hard_out: Indicates whether to output hard or soft decisions on
        the decoded information vector. `True` implies a hard-decoded
        information vector of 0/1's is output. `False` implies decoded LLRs
        of the information is output.
    :param algorithm: Indicates the implemented BCJR algorithm.
        `"map"` denotes the exact MAP algorithm, `"log"` indicates the
        exact MAP implementation, but in log-domain, and
        `"maxlog"` indicates the approximated MAP implementation in
        log-domain, where :math:`\log(e^{a}+e^{b}) \sim \max(a,b)`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: `torch.float`.
        Tensor of shape `[..., n]` containing the (noisy) channel
        output symbols where `n` is the codeword length.

    :output output: `torch.float`.
        Tensor of shape `[..., coderate * n]` containing the estimates of the
        information bit tensor.

    .. rubric:: Notes

    For decoding, input `logits` defined as
    :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}` are assumed for
    compatibility with the rest of Sionna. Internally,
    log-likelihood ratios (LLRs) with definition
    :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder

        encoder = TurboEncoder(rate=1/3, constraint_length=4, terminate=True)
        decoder = TurboDecoder(encoder, num_iter=6)

        u = torch.randint(0, 2, (10, 40), dtype=torch.float32)
        c = encoder(u)

        # Simulate BPSK with AWGN
        x = 2.0 * c - 1.0
        y = x + 0.5 * torch.randn_like(x)
        llr = 2.0 * y / 0.25

        u_hat = decoder(llr)
        print(u_hat.shape)
        # torch.Size([10, 40])
    """

    def __init__(
        self,
        encoder: Optional["TurboEncoder"] = None,
        gen_poly: Optional[tuple] = None,
        rate: float = 1 / 3,
        constraint_length: Optional[int] = None,
        interleaver: str = "3GPP",
        terminate: bool = False,
        num_iter: int = 6,
        hard_out: bool = True,
        algorithm: str = "map",
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if encoder is not None:
            self._coderate = encoder._coderate
            self._gen_poly = encoder._gen_poly
            self._terminate = encoder.terminate
            self._trellis = encoder.trellis
            if self._trellis._device != self.device:
                self._trellis.to(self.device)
            if not self._trellis.rsc:
                raise ValueError("Trellis must be RSC.")
            self._rsc = True
            self._internal_interleaver = encoder.internal_interleaver
        else:
            if gen_poly is not None:
                if not all(isinstance(poly, str) for poly in gen_poly):
                    raise TypeError("Each polynomial must be a string.")
                if not all(len(poly) == len(gen_poly[0]) for poly in gen_poly):
                    raise ValueError(
                        "Each polynomial must be of same length."
                    )
                if not all(
                    all(char in ["0", "1"] for char in poly)
                    for poly in gen_poly
                ):
                    raise ValueError(
                        "Each polynomial must be a string of 0's and 1's."
                    )
                self._gen_poly = gen_poly
            else:
                valid_constraint_length = (3, 4, 5, 6)
                if constraint_length not in valid_constraint_length:
                    raise ValueError(
                        "Constraint length must be between 3 and 6."
                    )
                self._gen_poly = polynomial_selector(constraint_length)

            valid_rates = (1 / 2, 1 / 3)
            if rate not in valid_rates:
                raise ValueError("rate must be 1/3 or 1/2.")
            self._coderate = rate

            if not isinstance(terminate, bool):
                raise TypeError("terminate must be bool.")
            self._terminate = terminate

            if interleaver not in ("3GPP", "random"):
                raise ValueError("interleaver must be 3GPP or random.")

            if interleaver == "3GPP":
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

            self._rsc = True
            self._trellis = Trellis(
                self._gen_poly, rsc=self._rsc, device=self.device
            )

        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be bool.")

        self._coderate_conv = 1 / len(self._gen_poly)
        self._mu = len(self._gen_poly[0]) - 1

        self._punct_pattern = puncture_pattern(
            self._coderate, self._coderate_conv, device=self.device
        )

        # Number of input bit streams, only 1 in current implementation
        self._conv_k = self._trellis.conv_k
        self._mu = self._trellis._mu
        # Number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n
        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        if self._conv_k != 1:
            raise NotImplementedError("Only single bit stream support.")
        if self._conv_n != 2:
            raise NotImplementedError("Only single bit stream support.")

        # For conv codes, the code dimensions are unknown during initialization
        self._k: Optional[int] = None  # Length of Info-bit vector
        self._n: Optional[int] = None  # Length of Turbo codeword

        if self._terminate:
            self._turbo_term = TurboTermination(
                self._mu + 1, conv_n=self._conv_n, device=self.device
            )
            self._num_term_bits = 3 * self._turbo_term.get_num_term_syms()
        else:
            self._turbo_term = None
            self._num_term_bits = 0

        self._num_iter = num_iter
        self._hard_out = hard_out

        self._bcjrdecoder = BCJRDecoder(
            gen_poly=self._gen_poly,
            rsc=self._rsc,
            hard_out=False,
            terminate=self._terminate,
            algorithm=algorithm,
            precision=precision,
            device=device,
        )

        # Internal state
        self.register_buffer("_punct_indices", None)
        self._depunct_len: Optional[int] = None
        self._convenc_numsyms: Optional[int] = None

    @property
    def gen_poly(self) -> tuple:
        """Generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def constraint_length(self) -> int:
        """Constraint length of the encoder."""
        return self._mu + 1

    @property
    def coderate(self) -> float:
        """Rate of the code used in the encoder."""
        return self._coderate

    @property
    def trellis(self) -> Trellis:
        """Trellis object used during encoding."""
        return self._trellis

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

    def depuncture(self, y: torch.Tensor) -> torch.Tensor:
        """Depuncture by scattering elements into a larger tensor with zeros.

        Given a tensor ``y`` of shape `[batch, n]`, scatters ``y`` elements
        into shape `[batch, 3*rate*n]` where the extra elements are filled
        with 0.

        For example, if input is ``y``, rate is 1/2 and ``punct_pattern`` is
        ``[1, 1, 0, 1, 0, 1]``, then the output is
        ``[y[0], y[1], 0., y[2], 0., y[3], y[4], y[5], 0., ... ,]``.

        :param y: Tensor of shape `[batch, n]` containing received LLRs.

        :output y_depunct: Depunctured tensor of shape `[batch, 3*rate*n]`.
        """
        batch_size = y.shape[0]
        input_device = y.device

        # Create output tensor filled with zeros
        y_depunct = torch.zeros(
            self._depunct_len,
            batch_size,
            dtype=y.dtype,
            device=input_device,
        )

        # Ensure punct_indices is on the correct device
        punct_indices = self._punct_indices
        if punct_indices.device != input_device:
            punct_indices = punct_indices.to(input_device)

        # Scatter values to punctured positions
        y_depunct[punct_indices.squeeze(-1)] = y.t()
        y_depunct = y_depunct.t()

        return y_depunct

    def _convenc_cws(
        self, y_turbo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-arrange Turbo codeword to two convolutional codewords format.

        Given the channel output of a Turbo codeword ``y_turbo``, this method
        re-arranges ``y_turbo`` such that ``y1_cw`` contains the symbols
        corresponding to convolutional encoder 1 and similarly ``y2_cw``
        contains the symbols corresponding to convolutional encoder 2.

        :param y_turbo: Channel output of a Turbo codeword.

        :output y1_cw: Symbols corresponding to convolutional encoder 1.

        :output y2_cw: Symbols corresponding to convolutional encoder 2.
        """
        input_device = y_turbo.device
        y_turbo = self.depuncture(y_turbo)
        prepunct_n = int(self._n * 3 * self._coderate)

        # Separate pre-termination & termination parts of Y
        y_cw = y_turbo[:, : prepunct_n - self._num_term_bits]
        y_term = y_turbo[:, prepunct_n - self._num_term_bits : prepunct_n]

        # Gather encoder 1 correspondence from Y (pre-termination part)
        enc1_sys_idx = torch.arange(
            0, self._k * 3, 3, device=input_device
        ).unsqueeze(1)

        # Systematic + parity from encoder 1
        enc1_cw_idx = torch.stack([enc1_sys_idx, enc1_sys_idx + 1], dim=1)
        enc1_cw_idx = enc1_cw_idx.reshape(-1)
        y1_cw = y_cw[:, enc1_cw_idx]

        # Gather systematic part from encoder 1 & inverse-interleave
        y1_sys_cw = y_cw[:, enc1_sys_idx.squeeze()]
        y2_sys_cw = self._internal_interleaver(y1_sys_cw).unsqueeze(-1)

        # Parity from encoder 2
        y2_nonsys_cw = y_cw[:, enc1_sys_idx.squeeze() + 2].unsqueeze(-1)

        # Stack systematic + parity for encoder 2
        y2_cw = torch.cat([y2_sys_cw, y2_nonsys_cw], dim=-1)
        y2_cw = y2_cw.reshape(-1, 2 * self._k)

        # Separate termination bits to encoders 1 & 2
        if self._terminate:
            term_vec1, term_vec2 = self._turbo_term.term_bits_turbo2conv(y_term)
            y1_cw = torch.cat([y1_cw, term_vec1], dim=1)
            y2_cw = torch.cat([y2_cw, term_vec2], dim=-1)

        return y1_cw, y2_cw

    def build(self, input_shape: tuple) -> None:
        """Build block and check dimensions.

        :param input_shape: Shape of input tensor [..., n].
        """
        self._n = input_shape[-1]

        if self.coderate == 1 / 2:
            if self._n % 2 != 0:
                raise ValueError("Codeword length should be a multiple of 2.")

        codefactor = self.coderate * 3
        turbo_n = int(self._n * codefactor)
        turbo_n_preterm = turbo_n - self._num_term_bits

        if turbo_n_preterm % 3 != 0:
            raise ValueError(
                "Invalid codeword length for a terminated Turbo code."
            )

        self._k = int(turbo_n_preterm / 3)

        # Number of symbols for the convolutional codes
        self._convenc_numsyms = self._k
        if self._terminate:
            self._convenc_numsyms += self._mu

        # Generate puncturing mask
        rate_factor = 3.0 * self._coderate
        self._depunct_len = int(rate_factor * self._n)

        punct_size = self._punct_pattern.numel()
        rep_times = self._depunct_len // punct_size

        mask_ = self._punct_pattern.repeat(rep_times, 1)
        extra_bits = self._depunct_len - rep_times * punct_size

        if extra_bits > 0:
            extra_periods = extra_bits // 3
            mask_ = torch.cat(
                [mask_, self._punct_pattern[:extra_periods, :]], dim=0
            )

        mask_ = mask_.reshape(-1)
        self.register_buffer("_punct_indices", torch.where(mask_)[0].unsqueeze(-1).to(torch.int32))

    @torch.compiler.disable
    def call(self, llr_ch: torch.Tensor, /) -> torch.Tensor:
        """Turbo decoding function.

        Runs BCJR decoder on both the constituent convolutional codes
        iteratively ``num_iter`` times. At the end, the resultant LLRs are
        computed and the decoded message vector (termination bits are
        excluded) is output.

        :param llr_ch: Channel LLRs of shape `[..., n]`.

        :output output: Decoded information tensor of shape
            `[..., coderate * n]`.

        .. rubric:: Notes

        This method uses ``@torch.compiler.disable`` because the iterative
        decoding loop and internal BCJRDecoder calls cause slow compilation
        with ``torch.compile``.

        .. rubric:: Examples


        .. code-block:: python

            import torch
            from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder

            encoder = TurboEncoder(rate=1/3, constraint_length=4, terminate=True)
            decoder = TurboDecoder(encoder, num_iter=6)

            u = torch.randint(0, 2, (10, 40), dtype=torch.float32)
            c = encoder(u)

            # Simulate BPSK with AWGN
            x = 2.0 * c - 1.0
            y = x + 0.5 * torch.randn_like(x)
            llr = 2.0 * y / 0.25

            u_hat = decoder(llr)
            print(u_hat.shape)
            # torch.Size([10, 40])
        """
        llr_max = 20.0
        input_device = llr_ch.device
        input_dtype = llr_ch.dtype

        output_shape = list(llr_ch.shape)

        # Allow different codeword lengths in eager mode
        if output_shape[-1] != self._n:
            self._built = False
            self.build(llr_ch.shape)
            self._built = True

        llr_ch = llr_ch.reshape(-1, self._n)
        batch_size = llr_ch.shape[0]

        output_shape[-1] = self._k

        # Get codewords for each encoder
        y1_cw, y2_cw = self._convenc_cws(llr_ch)

        # Extract systematic LLRs
        sys_idx = torch.arange(0, self._k * 2, 2, device=input_device)
        llr_ch_sys = y1_cw[:, sys_idx]
        llr_ch2_sys = y2_cw[:, sys_idx]

        llr_1e = torch.zeros(
            batch_size, self._convenc_numsyms, dtype=input_dtype, device=input_device
        )

        # Define zero LLRs for termination info bits
        term_info_bits = self._mu if self._terminate else 0
        llr_terminfo = torch.zeros(
            batch_size, term_info_bits, dtype=input_dtype, device=input_device
        )

        # Needs to be initialized before entering the loop
        llr_2i = torch.zeros_like(llr_ch2_sys)

        # Run decoding loop
        for _ in range(self._num_iter):
            # Run 1st component decoder
            llr_1i = self._bcjrdecoder(y1_cw, llr_a=llr_1e)
            llr_1i = llr_1i[..., : self._k]
            llr_extr = llr_1i - llr_ch_sys - llr_1e[..., : self._k]

            llr_2e = self._internal_interleaver(llr_extr)
            llr_2e = torch.cat([llr_2e, llr_terminfo], dim=-1)
            llr_2e = torch.clamp(llr_2e, min=-llr_max, max=llr_max)

            # Run 2nd component decoder
            llr_2i = self._bcjrdecoder(y2_cw, llr_a=llr_2e)
            llr_2i = llr_2i[..., : self._k]
            llr_extr = llr_2i - llr_2e[..., : self._k] - llr_ch2_sys

            llr_1e = self._internal_interleaver(llr_extr, inverse=True)
            llr_1e = torch.clamp(llr_1e, min=-llr_max, max=llr_max)
            llr_1e = torch.cat([llr_1e, llr_terminfo], dim=-1)

        # Use latest output of 2nd decoder
        output = self._internal_interleaver(llr_2i, inverse=True)

        if self._hard_out:
            output = (output > 0.0).to(self.dtype)
        else:
            output = output.to(self.dtype)

        output_reshaped = output.reshape(output_shape)
        return output_reshaped

