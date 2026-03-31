#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for Polar decoding such as successive cancellation (SC),
successive cancellation list (SCL) and iterative belief propagation (BP)
decoding."""

from typing import Optional, Tuple, Union
import numbers
import warnings
import numpy as np
import torch
import torch.nn.functional as F

from sionna.phy import Block
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.polar.encoding import Polar5GEncoder


__all__ = [
    "PolarSCDecoder",
    "PolarSCLDecoder",
    "PolarBPDecoder",
    "Polar5GDecoder",
]


class PolarSCDecoder(Block):
    """Successive cancellation (SC) decoder :cite:p:`Arikan_Polar` for Polar codes
    and Polar-like codes.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel LLR values (as logits).

    :output u_hat: [..., k], `torch.float`.
        Tensor containing hard-decided estimations of all ``k``
        information bits.

    .. rubric:: Notes

    This block implements the SC decoder as described in
    :cite:p:`Arikan_Polar`. However, the implementation follows the `recursive
    tree` :cite:p:`Gross_Fast_SCL` terminology and combines nodes for increased
    throughputs without changing the outcome of the algorithm.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarSCDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarSCDecoder(frozen_pos, n)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)

        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            msg = "Internal error: invalid info_pos generated."
            raise ArithmeticError(msg)

        # Register info_pos as buffer for torch.compile compatibility
        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )

        self._llr_max = 30.0  # Internal max LLR value
        # Create a frozen bit vector for simpler encoding
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1

        # Register frozen indicator as tensor buffer for torch.compile compatibility
        self.register_buffer(
            "_frozen_ind_t",
            torch.tensor(self._frozen_ind, dtype=self.dtype, device=self.device),
        )

        # Enable graph pruning
        self._use_fast_sc = False

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self) -> np.ndarray:
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self) -> np.ndarray:
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    def _cn_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.

        See :cite:p:`Stimming_LLR` and :cite:p:`Hashemi_SSCL` for detailed equations.
        """
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        # Avoid division for numerical stability
        llr_out = torch.log(1 + torch.exp(x_in + y_in))
        llr_out = llr_out - torch.log(torch.exp(x_in) + torch.exp(y_in))

        return llr_out

    def _vn_op(
        self, x: torch.Tensor, y: torch.Tensor, u_hat: torch.Tensor
    ) -> torch.Tensor:
        """VN update for LLR inputs."""
        return (1 - 2 * u_hat) * x + y

    def _polar_decode_sc(
        self, llr_ch: torch.Tensor, frozen_ind: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recursive SC decoding function.

        Recursively branch decoding tree and split into decoding of `upper`
        and `lower` path until reaching a leaf node.

        The function returns the u_hat decisions at stage `0` and the bit
        decisions of the intermediate stage `s` (i.e., the re-encoded
        version of `u_hat` until the current stage `s`).

        This decoder parallelizes over the batch-dimension, i.e., the tree
        is processed for all samples in the batch in parallel. This yields a
        higher throughput, but does not improve the latency.
        """
        # Calculate current codeword length
        n = frozen_ind.shape[0]

        # Branch if leaf is not reached yet
        if n > 1:
            if self._use_fast_sc:
                if frozen_ind.sum() == n:
                    u_hat = torch.zeros_like(llr_ch)
                    return u_hat, u_hat

            llr_ch1 = llr_ch[..., 0 : int(n / 2)]
            llr_ch2 = llr_ch[..., int(n / 2) :]
            frozen_ind1 = frozen_ind[0 : int(n / 2)]
            frozen_ind2 = frozen_ind[int(n / 2) :]

            # Upper path
            x_llr1_in = self._cn_op(llr_ch1, llr_ch2)

            # Call the decoding function (with upper half)
            u_hat1, u_hat1_up = self._polar_decode_sc(x_llr1_in, frozen_ind1)

            # Lower path
            x_llr2_in = self._vn_op(llr_ch1, llr_ch2, u_hat1_up)
            # Call the decoding function again (with lower half)
            u_hat2, u_hat2_up = self._polar_decode_sc(x_llr2_in, frozen_ind2)

            # Combine u_hat from both branches
            u_hat = torch.cat([u_hat1, u_hat2], -1)

            # Calculate re-encoded version of u_hat at current stage
            u_hat1_up_int = u_hat1_up.to(torch.int8)
            u_hat2_up_int = u_hat2_up.to(torch.int8)
            u_hat1_up_int = torch.bitwise_xor(u_hat1_up_int, u_hat2_up_int)
            u_hat1_up = u_hat1_up_int.to(self.dtype)
            u_hat_up = torch.cat([u_hat1_up, u_hat2_up], -1)

        else:  # If leaf is reached perform basic decoding op (=decision)
            # Use tensor operations to avoid CUDA graph breaks
            # frozen_ind is a 1-element tensor at this point
            is_frozen = frozen_ind[0] == 1  # Tensor comparison

            # Compute frozen case: u_hat = 0
            frozen_result = torch.zeros_like(llr_ch)

            # Compute non-frozen case: hard decision
            decision_result = 0.5 * (1.0 - torch.sign(llr_ch))
            # Handle exact 0 LLRs (u_hat = 0.5) by setting to 1
            decision_result = torch.where(
                decision_result == 0.5,
                torch.ones_like(decision_result),
                decision_result,
            )

            # Branchless selection using torch.where
            u_hat = torch.where(is_frozen, frozen_result, decision_result)
            u_hat_up = u_hat
        return u_hat, u_hat_up

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Check if shape of input is invalid."""
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(self, llr_ch: torch.Tensor) -> torch.Tensor:
        """Successive cancellation (SC) decoding function.

        Performs successive cancellation decoding and returns the estimated
        information bits.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel LLR values (as logits).

        :output u_hat: Tensor of shape `[..., k]` containing hard-decided
            estimations of all ``k`` information bits.

        Note: This function recursively unrolls the SC decoding tree, thus,
        for larger values of ``n`` building the decoding graph can become
        time consuming.
        """
        # Reshape inputs to [-1, n]
        input_shape = llr_ch.shape
        new_shape = (-1, self._n)
        llr_ch = llr_ch.reshape(new_shape)

        llr_ch = -1.0 * llr_ch  # Logits are converted into "true" llrs

        # Decode
        u_hat_n, _ = self._polar_decode_sc(llr_ch, self._frozen_ind_t)

        # Recover the k information bit positions using pre-registered buffer
        u_hat = u_hat_n[:, self._info_pos_t]

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self.k]
        u_hat_reshape = u_hat.reshape(output_shape)
        return u_hat_reshape


class PolarSCLDecoder(Block):
    # pylint: disable=line-too-long
    """Successive cancellation list (SCL) decoder :cite:p:`Tal_SCL` for Polar codes
    and Polar-like codes.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param list_size: Defines the list size of the decoder.
    :param crc_degree: Defining the CRC polynomial to be used. Can be any
        value from `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
    :param use_hybrid_sc: If `True`, SC decoding is applied and only the
        codewords with invalid CRC are decoded with SCL. This option
        requires an outer CRC specified via ``crc_degree``.
    :param use_fast_scl: If `True`, tree pruning is used to reduce
        the decoding complexity. The output is equivalent to the
        non-pruned version (besides numerical differences).
    :param cpu_only: If `True`, a NumPy-based decoder runs on the CPU.
        This option is usually slower, but also more memory efficient
        and in particular recommended for larger blocklengths.
    :param use_scatter: If `True`, scatter update is used for tensor
        updates. This option is usually slower, but more memory efficient.
    :param ind_iil_inv: If not `None`, the sequence is used as inverse
        input bit interleaver before evaluating the CRC. This only
        affects the CRC evaluation but the output sequence is not
        permuted.
    :param return_crc_status: If `True`, the decoder additionally returns
        the CRC status indicating if a codeword was (most likely)
        correctly recovered. This is only available if ``crc_degree`` is
        not `None`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel LLR values (as logits).

    :output b_hat: [..., k], `torch.float`.
        Binary tensor containing hard-decided estimations of all `k`
        information bits.

    :output crc_status: [...], `torch.bool`.
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if ``return_crc_status`` is `True`.
        Note that false positives are possible.

    .. rubric:: Notes

    This block implements the successive cancellation list (SCL) decoder
    as described in :cite:p:`Tal_SCL` but uses LLR-based message updates
    :cite:p:`Stimming_LLR`. The implementation follows the notation from
    :cite:p:`Gross_Fast_SCL`, :cite:p:`Hashemi_SSCL`. If option ``use_fast_scl`` is
    active, tree pruning is used and tree nodes are combined if possible
    (see :cite:p:`Hashemi_SSCL` for details).

    For longer code lengths, the complexity of the decoding graph becomes
    large and we recommend to use the ``cpu_only`` option that uses an
    embedded NumPy decoder. Further, this function recursively unrolls the
    SCL decoding tree, thus, for larger values of ``n`` building the
    decoding graph can become time consuming. Please consider the
    ``cpu_only`` option if building the graph takes too long.

    A hybrid SC/SCL decoder as proposed in :cite:p:`Cammerer_Hybrid_SCL` (using
    SC instead of BP) can be activated with option ``use_hybrid_sc`` iff
    an outer CRC is available. Please note that the results are not
    exactly SCL performance caused by the false positive rate of the CRC.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarSCLDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarSCLDecoder(frozen_pos, n, list_size=8)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        list_size: int = 8,
        crc_degree: Optional[str] = None,
        use_hybrid_sc: bool = False,
        use_fast_scl: bool = True,
        cpu_only: bool = False,
        use_scatter: bool = False,
        ind_iil_inv: Optional[np.ndarray] = None,
        return_crc_status: bool = False,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if not isinstance(list_size, int):
            raise TypeError("list_size must be integer.")
        if not isinstance(cpu_only, bool):
            raise TypeError("cpu_only must be bool.")
        if not isinstance(use_scatter, bool):
            raise TypeError("use_scatter must be bool.")
        if not isinstance(use_fast_scl, bool):
            raise TypeError("use_fast_scl must be bool.")
        if not isinstance(use_hybrid_sc, bool):
            raise TypeError("use_hybrid_sc must be bool.")
        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")

        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")
        if np.log2(list_size) != int(np.log2(list_size)):
            raise ValueError("list_size must be a power of 2.")

        # CPU mode is recommended for larger values of n
        if n > 128 and cpu_only is False and use_hybrid_sc is False:
            warnings.warn(
                "Required resource allocation is large "
                "for the selected blocklength. Consider option `cpu_only=True`."
            )

        # CPU mode is recommended for larger values of L
        if list_size > 32 and cpu_only is False and use_hybrid_sc is False:
            warnings.warn(
                "Resource allocation is high for the "
                "selected list_size. Consider option `cpu_only=True`."
            )

        # Internal decoder parameters
        self._use_fast_scl = use_fast_scl
        self._use_scatter = use_scatter
        self._cpu_only = cpu_only
        self._use_hybrid_sc = use_hybrid_sc

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._list_size = list_size
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        self._llr_max = 30.0
        if self._k != len(self._info_pos):
            raise ArithmeticError("Internal error: invalid info_pos generated.")

        # Create a frozen bit vector
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1
        self._cw_ind = np.arange(self._n)
        self._n_stages = int(np.log2(self._n))

        # Register frozen indicator as tensor buffer for torch.compile compatibility
        self.register_buffer(
            "_frozen_ind_t",
            torch.tensor(self._frozen_ind, dtype=self.dtype, device=self.device),
        )

        # Register info_pos as tensor buffer to avoid torch.tensor() in call
        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )

        # Init CRC check (if needed)
        if crc_degree is not None:
            self._use_crc = True
            self._crc_encoder = CRCEncoder(
                crc_degree, precision=precision, device=device
            )
            self._crc_decoder = CRCDecoder(
                self._crc_encoder, precision=precision, device=device
            )
            self._k_crc = self._crc_decoder.encoder.crc_length
        else:
            self._use_crc = False
            self._k_crc = 0
        if self._k < self._k_crc:
            msg = "Value of k is too small for given CRC_degree."
            raise ValueError(msg)

        if (crc_degree is None) and return_crc_status:
            self._return_crc_status = False
            raise ValueError("Returning CRC status requires given crc_degree.")
        else:
            self._return_crc_status = return_crc_status

        # Store the inverse interleaver pattern
        if ind_iil_inv is not None:
            if ind_iil_inv.shape[0] != self._k:
                raise ValueError("ind_int must be of length k+k_crc.")
            self._ind_iil_inv = ind_iil_inv
            self._iil = True
            # Register as tensor buffer to avoid torch.tensor() in call
            self.register_buffer(
                "_ind_iil_inv_t",
                torch.tensor(ind_iil_inv, dtype=torch.int32, device=self.device),
            )
        else:
            self._iil = False

        # Use SC decoder first and use numpy-based SCL as "afterburner"
        if self._use_hybrid_sc:
            self._decoder_sc = PolarSCDecoder(
                frozen_pos, n, precision=precision, device=device
            )
            if not self._use_crc:
                raise ValueError("Hybrid SC requires outer CRC.")

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self._k

    @property
    def k_crc(self) -> int:
        """Number of CRC bits."""
        return self._k_crc

    @property
    def frozen_pos(self) -> np.ndarray:
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self) -> np.ndarray:
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def list_size(self) -> int:
        """List size for SCL decoding."""
        return self._list_size

    # NumPy-based decoder helper functions

    def _cn_op_np(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check node update (boxplus) for LLRs in NumPy.

        See :cite:p:`Stimming_LLR` and :cite:p:`Hashemi_SSCL` for detailed equations.
        """
        x_in = np.maximum(np.minimum(x, self._llr_max), -self._llr_max)
        y_in = np.maximum(np.minimum(y, self._llr_max), -self._llr_max)

        llr_out = np.log(1 + np.exp(x_in + y_in))
        llr_out -= np.log(np.exp(x_in) + np.exp(y_in))

        return llr_out

    def _vn_op_np(
        self, x: np.ndarray, y: np.ndarray, u_hat: np.ndarray
    ) -> np.ndarray:
        """Variable node update (boxplus) for LLRs in Numpy."""
        return np.multiply((1 - 2 * u_hat), x) + y

    def _update_rate0_code_np(self, cw_ind: np.ndarray) -> None:
        """Update rate-0 (i.e., all frozen) sub-code at pos ``cw_ind``.

        See Eq. (26) in :cite:p:`Hashemi_SSCL`.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        ind = np.expand_dims(self._dec_pointer, axis=-1)
        llr_in = np.take_along_axis(
            self.msg_llr[:, :, stage_ind, cw_ind], ind, axis=1
        )

        llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
        pm_val = np.log(1 + np.exp(-llr_clip))
        self.msg_pm += np.sum(pm_val, axis=-1)

    def _update_rep_code_np(self, cw_ind: np.ndarray) -> None:
        """Update rep. code sub-code at position ``cw_ind``.

        See Eq. (31) in :cite:p:`Hashemi_SSCL`.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))
        bs = self._dec_pointer.shape[0]

        llr = np.zeros([bs, 2 * self._list_size, n])
        for i in range(bs):
            llr_i = self.msg_llr[i, self._dec_pointer[i, :], stage_ind, :]
            llr[i, :, :] = llr_i[:, cw_ind]

        llr[:, self._list_size :, :] = -llr[:, self._list_size :, :]
        llr_in = np.maximum(np.minimum(llr, self._llr_max), -self._llr_max)
        pm_val = np.sum(np.log(1 + np.exp(-llr_in)), axis=-1)
        self.msg_pm += pm_val

        for i in range(bs):
            ind_dec = self._dec_pointer[i, self._list_size :]
            for j in cw_ind:
                self.msg_uhat[i, ind_dec, stage_ind, j] = 1

        self._update_single_bit_np([cw_ind[-1]])
        self._sort_decoders_np()
        self._duplicate_paths_np()

    def _update_single_bit_np(self, ind_u: list) -> None:
        """Update single bit at position ``ind_u`` of all decoders."""
        if self._frozen_ind[ind_u] == 0:
            ind_dec = np.expand_dims(
                self._dec_pointer[:, self._list_size :], axis=-1
            )
            uhat_slice = self.msg_uhat[:, :, 0, ind_u]
            np.put_along_axis(uhat_slice, ind_dec, 1.0, axis=1)
            self.msg_uhat[:, :, 0, ind_u] = uhat_slice

    def _update_pm_np(self, ind_u: list) -> None:
        """Update path metric of all decoders at bit position ``ind_u``.

        We apply Eq. (10) from :cite:p:`Stimming_LLR`.
        """
        ind = np.expand_dims(self._dec_pointer, axis=-1)
        u_hat = np.take_along_axis(self.msg_uhat[:, :, 0, ind_u], ind, axis=1)
        u_hat = np.squeeze(u_hat, axis=-1)
        llr_in = np.take_along_axis(self.msg_llr[:, :, 0, ind_u], ind, axis=1)
        llr_in = np.squeeze(llr_in, axis=-1)

        llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
        self.msg_pm += np.log(
            1 + np.exp(-np.multiply((1 - 2 * u_hat), llr_clip))
        )

    def _sort_decoders_np(self) -> None:
        """Sort decoders according to their path metric."""
        ind = np.argsort(self.msg_pm, axis=-1)
        self.msg_pm = np.take_along_axis(self.msg_pm, ind, axis=1)
        self._dec_pointer = np.take_along_axis(self._dec_pointer, ind, axis=1)

    def _duplicate_paths_np(self) -> None:
        """Copy first ``list_size``/2 paths into lower part.

        Decoder indices are encoded in ``self._dec_pointer``.
        """
        ind_low = self._dec_pointer[:, : self._list_size]
        ind_up = self._dec_pointer[:, self._list_size :]

        for i in range(ind_up.shape[0]):
            self.msg_uhat[i, ind_up[i, :], :, :] = self.msg_uhat[
                i, ind_low[i, :], :, :
            ]
            self.msg_llr[i, ind_up[i, :], :, :] = self.msg_llr[
                i, ind_low[i, :], :, :
            ]

        self.msg_pm[:, self._list_size :] = self.msg_pm[:, : self._list_size]

    def _polar_decode_scl_np(self, cw_ind: np.ndarray) -> None:
        """Recursive decoding function in NumPy.

        We follow the terminology from :cite:p:`Hashemi_SSCL` and
        :cite:p:`Stimming_LLR` and branch the messages into a `left` and `right`
        update paths until reaching a leaf node.

        Tree pruning as proposed in :cite:p:`Hashemi_SSCL` is used to minimize
        the tree depth while maintaining the same output.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        if n > 1:
            if self._use_fast_scl:
                if np.sum(self._frozen_ind[cw_ind]) == n:
                    self._update_rate0_code_np(cw_ind)
                    return
                if (
                    self._frozen_ind[cw_ind[-1]] == 0
                    and np.sum(self._frozen_ind[cw_ind[:-1]]) == n - 1
                ):
                    self._update_rep_code_np(cw_ind)
                    return

            cw_ind_left = cw_ind[0 : int(n / 2)]
            cw_ind_right = cw_ind[int(n / 2) :]

            # Left branch
            llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
            llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]

            self.msg_llr[:, :, stage_ind - 1, cw_ind_left] = self._cn_op_np(
                llr_left, llr_right
            )

            self._polar_decode_scl_np(cw_ind_left)

            # Right branch
            u_hat_left_up = self.msg_uhat[:, :, stage_ind - 1, cw_ind_left]
            llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
            llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]

            self.msg_llr[:, :, stage_ind - 1, cw_ind_right] = self._vn_op_np(
                llr_left, llr_right, u_hat_left_up
            )

            self._polar_decode_scl_np(cw_ind_right)

            # Combine u_hat
            u_hat_left_up = self.msg_uhat[:, :, stage_ind - 1, cw_ind_left]
            u_hat_right_up = self.msg_uhat[:, :, stage_ind - 1, cw_ind_right]

            u_hat_left = (u_hat_left_up != u_hat_right_up) + 0
            u_hat = np.concatenate([u_hat_left, u_hat_right_up], axis=-1)

            self.msg_uhat[:, :, stage_ind, cw_ind] = u_hat

        else:
            self._update_single_bit_np(cw_ind)
            self._update_pm_np(cw_ind)

            if self._frozen_ind[cw_ind] == 0:
                self._sort_decoders_np()
                self._duplicate_paths_np()

    def _decode_np_batch(
        self, llr_ch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode batch of ``llr_ch`` with Numpy decoder."""
        bs = llr_ch.shape[0]

        self.msg_uhat = np.zeros(
            [bs, 2 * self._list_size, self._n_stages + 1, self._n]
        )
        self.msg_llr = np.zeros(
            [bs, 2 * self._list_size, self._n_stages + 1, self._n]
        )
        self.msg_pm = np.zeros([bs, 2 * self._list_size])

        self.msg_pm[:, 1 : self._list_size] = self._llr_max
        self.msg_pm[:, self._list_size + 1 :] = self._llr_max

        self._dec_pointer = np.arange(2 * self._list_size)
        self._dec_pointer = np.tile(
            np.expand_dims(self._dec_pointer, axis=0), [bs, 1]
        )

        self.msg_llr[:, :, self._n_stages, :] = np.expand_dims(llr_ch, axis=1)

        self._polar_decode_scl_np(self._cw_ind)

        self._sort_decoders_np()

        for ind in range(bs):
            self.msg_uhat[ind, :, :, :] = self.msg_uhat[
                ind, self._dec_pointer[ind], :, :
            ]
        return self.msg_uhat, self.msg_pm

    def _decode_np_hybrid(
        self,
        llr_ch: np.ndarray,
        u_hat_sc: np.ndarray,
        crc_valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid SCL decoding stage that decodes iff CRC from previous SC
        decoding attempt failed.

        This option avoids the usage of the high-complexity SCL decoder in
        cases where SC would be sufficient. For further details we refer to
        :cite:p:`Cammerer_Hybrid_SCL` (we use SC instead of the proposed BP
        stage).

        This decoder does not exactly implement SCL as the CRC can be
        false positive after the SC stage. However, in these cases SCL+CRC
        may also yield the wrong results.
        """
        bs = llr_ch.shape[0]
        crc_valid = np.squeeze(crc_valid, axis=-1)
        ind_invalid = np.arange(bs)[np.invert(crc_valid)]

        llr_ch_hyb = np.take(llr_ch, ind_invalid, axis=0)
        msg_uhat_hyb, msg_pm_hyb = self._decode_np_batch(llr_ch_hyb)

        msg_uhat = np.zeros([bs, 2 * self._list_size, 1, self._n])
        msg_pm = np.ones([bs, 2 * self._list_size]) * self._llr_max * self.k
        msg_pm[:, 0] = 0

        msg_uhat[:, 0, 0, self._info_pos] = u_hat_sc

        ind_hyb = 0
        for ind in range(bs):
            if not crc_valid[ind]:
                msg_uhat[ind, :, 0, :] = msg_uhat_hyb[ind_hyb, :, 0, :]
                msg_pm[ind, :] = msg_pm_hyb[ind_hyb, :]
                ind_hyb += 1

        return msg_uhat, msg_pm

    # =========================================================================
    # PyTorch tensor-based decoder helper functions (for GPU acceleration)
    # =========================================================================

    def _cn_op_pt(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Check-node update (boxplus) for LLR inputs in PyTorch.

        Operations are performed element-wise.
        See :cite:p:`Stimming_LLR` and :cite:p:`Hashemi_SSCL` for detailed equations.
        """
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        # Implements log(1+e^(x+y)) - log(e^x+e^y)
        llr_out = F.softplus(x_in + y_in)
        llr_out = llr_out - torch.logsumexp(
            torch.stack([x_in, y_in], dim=-1), dim=-1
        )
        return llr_out

    def _vn_op_pt(
        self, x: torch.Tensor, y: torch.Tensor, u_hat: torch.Tensor
    ) -> torch.Tensor:
        """Variable node update for LLR inputs in PyTorch.

        Operations are performed element-wise.

        See :cite:p:`Stimming_LLR` and :cite:p:`Hashemi_SSCL` for detailed equations.
        """
        return (1 - 2 * u_hat) * x + y

    def _sort_decoders_pt(
        self,
        msg_pm: torch.Tensor,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort decoders according to their path metric in PyTorch."""
        ind = torch.argsort(msg_pm, dim=-1)

        # For msg_pm: [batch, 2*L]
        msg_pm = torch.gather(msg_pm, 1, ind)

        # For msg_uhat: [batch, 2*L, stages+1, n]
        ind_expanded = ind.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, msg_uhat.shape[2], msg_uhat.shape[3]
        )
        msg_uhat = torch.gather(msg_uhat, 1, ind_expanded)

        # For msg_llr: [batch, 2*L, stages+1, n]
        ind_expanded = ind.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, msg_llr.shape[2], msg_llr.shape[3]
        )
        msg_llr = torch.gather(msg_llr, 1, ind_expanded)

        return msg_pm, msg_uhat, msg_llr

    def _duplicate_paths_pt(
        self,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
        msg_pm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Duplicate paths by copying the upper branch into the lower one."""
        # Take first list_size paths and tile them
        msg_uhat = msg_uhat[:, : self._list_size, :, :].repeat(1, 2, 1, 1)
        msg_llr = msg_llr[:, : self._list_size, :, :].repeat(1, 2, 1, 1)
        msg_pm = msg_pm[:, : self._list_size].repeat(1, 2)
        return msg_uhat, msg_llr, msg_pm

    def _update_pm_pt(
        self,
        ind_u: np.ndarray,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
        msg_pm: torch.Tensor,
    ) -> torch.Tensor:
        """Update path metric after updating bit_pos ``ind_u`` in PyTorch.

        We implement Eq. (10) from :cite:p:`Stimming_LLR`.
        """
        u_hat = msg_uhat[:, :, 0, ind_u[0]]
        llr = msg_llr[:, :, 0, ind_u[0]]

        llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

        # Numerically stable: log(1 + exp(-x))
        msg_pm = msg_pm + F.softplus(-(1 - 2 * u_hat) * llr_in)
        return msg_pm

    def _update_single_bit_pt(
        self, ind_u: np.ndarray, msg_uhat: torch.Tensor
    ) -> torch.Tensor:
        """Update single bit at position ``ind_u`` for all decoders in PyTorch.

        Uses branchless computation for torch.compile compatibility.
        For info bits (non-frozen), sets upper half decoders' bit to 1.
        For frozen bits, sets to 0 (no-op since frozen bits are 0).
        """
        # Get info bit indicator from tensor buffer (1 if info, 0 if frozen)
        # This avoids data-dependent Python control flow
        is_info_bit = 1.0 - self._frozen_ind_t[ind_u[0]]

        # Set upper half decoders' bit at position ind_u
        # Value is 1 for info bits, 0 for frozen bits (branchless)
        msg_uhat1 = msg_uhat[:, : self._list_size, :, :]
        msg_uhat21 = msg_uhat[:, self._list_size :, 0:1, : ind_u[0]]
        msg_uhat22 = msg_uhat[:, self._list_size :, 0:1, ind_u[0] + 1 :]

        # Insert value: 1 if info bit, 0 if frozen
        batch_size = msg_uhat.shape[0]
        msg_insert = is_info_bit * torch.ones(
            batch_size, self._list_size, 1, 1,
            dtype=msg_uhat.dtype, device=msg_uhat.device
        )

        msg_uhat23 = torch.cat([msg_uhat21, msg_insert, msg_uhat22], dim=3)
        msg_uhat24 = msg_uhat[:, self._list_size :, 1:, :]

        msg_uhat2 = torch.cat([msg_uhat23, msg_uhat24], dim=2)
        msg_uhat = torch.cat([msg_uhat1, msg_uhat2], dim=1)

        return msg_uhat

    def _update_rate0_code_pt(
        self,
        msg_pm: torch.Tensor,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
        cw_ind: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update rate-0 sub-code (all frozen) at pos ``cw_ind`` in PyTorch.

        See Eq. (26) in :cite:p:`Hashemi_SSCL`.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        llr = msg_llr[:, :, stage_ind, cw_ind[0] : cw_ind[-1] + 1]
        llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

        # Update path metric for complete sub-block
        pm_val = F.softplus(-llr_in)
        msg_pm = msg_pm + pm_val.sum(dim=-1)

        return msg_pm, msg_uhat, msg_llr

    def _update_rep_code_pt(
        self,
        msg_pm: torch.Tensor,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
        cw_ind: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update rep. code sub-code at position ``cw_ind`` in PyTorch.

        See Eq. (31) in :cite:p:`Hashemi_SSCL`.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        # Get LLRs for this sub-code
        llr = msg_llr[:, :, stage_ind, cw_ind[0] : cw_ind[-1] + 1]
        llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

        # Upper branch has negative LLR values (bit is 1)
        llr_low = llr_in[:, : self._list_size, :]
        llr_up = -llr_in[:, self._list_size :, :]
        llr_pm = torch.cat([llr_low, llr_up], dim=1)

        pm_val = F.softplus(-llr_pm)
        msg_pm = msg_pm + pm_val.sum(dim=-1)

        # Set bits to 1 for upper branch decoders
        # Use split/concat approach
        msg_uhat1 = msg_uhat[:, : self._list_size, :, :]

        msg_uhat21 = msg_uhat[:, self._list_size :, stage_ind : stage_ind + 1, : cw_ind[0]]
        msg_uhat22 = msg_uhat[:, self._list_size :, stage_ind : stage_ind + 1, cw_ind[-1] + 1 :]

        batch_size = msg_uhat.shape[0]
        msg_ones = torch.ones(
            batch_size, self._list_size, 1, n,
            dtype=msg_uhat.dtype, device=msg_uhat.device
        )

        msg_uhat23 = torch.cat([msg_uhat21, msg_ones, msg_uhat22], dim=3)
        msg_uhat24_1 = msg_uhat[:, self._list_size :, :stage_ind, :]
        msg_uhat24_2 = msg_uhat[:, self._list_size :, stage_ind + 1 :, :]

        msg_uhat2 = torch.cat([msg_uhat24_1, msg_uhat23, msg_uhat24_2], dim=2)
        msg_uhat = torch.cat([msg_uhat1, msg_uhat2], dim=1)

        # Branch last bit and update
        msg_uhat = self._update_single_bit_pt([cw_ind[-1]], msg_uhat)
        msg_pm, msg_uhat, msg_llr = self._sort_decoders_pt(msg_pm, msg_uhat, msg_llr)
        msg_uhat, msg_llr, msg_pm = self._duplicate_paths_pt(msg_uhat, msg_llr, msg_pm)

        return msg_pm, msg_uhat, msg_llr

    def _update_left_branch_pt(
        self,
        msg_llr: torch.Tensor,
        stage_ind: int,
        cw_ind_left: np.ndarray,
        cw_ind_right: np.ndarray,
    ) -> torch.Tensor:
        """Update messages of left branch in PyTorch."""
        llr_left_in = msg_llr[:, :, stage_ind, cw_ind_left[0] : cw_ind_left[-1] + 1]
        llr_right_in = msg_llr[:, :, stage_ind, cw_ind_right[0] : cw_ind_right[-1] + 1]

        llr_left_out = self._cn_op_pt(llr_left_in, llr_right_in)

        # Use split/concatenation approach
        llr_left0 = msg_llr[:, :, stage_ind - 1, : cw_ind_left[0]]
        llr_right = msg_llr[:, :, stage_ind - 1, cw_ind_right[0] : cw_ind_right[-1] + 1]
        llr_right1 = msg_llr[:, :, stage_ind - 1, cw_ind_right[-1] + 1 :]

        llr_s = torch.cat([llr_left0, llr_left_out, llr_right, llr_right1], dim=2)
        llr_s = llr_s.unsqueeze(2)

        msg_llr1 = msg_llr[:, :, : stage_ind - 1, :]
        msg_llr2 = msg_llr[:, :, stage_ind:, :]
        msg_llr = torch.cat([msg_llr1, llr_s, msg_llr2], dim=2)

        return msg_llr

    def _update_right_branch_pt(
        self,
        msg_llr: torch.Tensor,
        msg_uhat: torch.Tensor,
        stage_ind: int,
        cw_ind_left: np.ndarray,
        cw_ind_right: np.ndarray,
    ) -> torch.Tensor:
        """Update messages for right branch in PyTorch."""
        u_hat_left_up = msg_uhat[:, :, stage_ind - 1, cw_ind_left[0] : cw_ind_left[-1] + 1]
        llr_left_in = msg_llr[:, :, stage_ind, cw_ind_left[0] : cw_ind_left[-1] + 1]
        llr_right = msg_llr[:, :, stage_ind, cw_ind_right[0] : cw_ind_right[-1] + 1]

        llr_right_out = self._vn_op_pt(llr_left_in, llr_right, u_hat_left_up)

        # Use split/concatenation approach
        llr_left0 = msg_llr[:, :, stage_ind - 1, : cw_ind_left[0]]
        llr_left = msg_llr[:, :, stage_ind - 1, cw_ind_left[0] : cw_ind_left[-1] + 1]
        llr_right1 = msg_llr[:, :, stage_ind - 1, cw_ind_right[-1] + 1 :]

        llr_s = torch.cat([llr_left0, llr_left, llr_right_out, llr_right1], dim=2)
        llr_s = llr_s.unsqueeze(2)

        msg_llr1 = msg_llr[:, :, : stage_ind - 1, :]
        msg_llr2 = msg_llr[:, :, stage_ind:, :]
        msg_llr = torch.cat([msg_llr1, llr_s, msg_llr2], dim=2)

        return msg_llr

    def _update_branch_u_pt(
        self,
        msg_uhat: torch.Tensor,
        stage_ind: int,
        cw_ind_left: np.ndarray,
        cw_ind_right: np.ndarray,
    ) -> torch.Tensor:
        """Update ``u_hat`` messages after executing both branches in PyTorch."""
        u_hat_left_up = msg_uhat[:, :, stage_ind - 1, cw_ind_left[0] : cw_ind_left[-1] + 1]
        u_hat_right_up = msg_uhat[:, :, stage_ind - 1, cw_ind_right[0] : cw_ind_right[-1] + 1]

        # Combine u_hat via XOR
        u_hat_left = (u_hat_left_up.int() ^ u_hat_right_up.int()).to(msg_uhat.dtype)

        # Use split/concatenation approach
        u_hat_left_0 = msg_uhat[:, :, stage_ind, : cw_ind_left[0]]
        u_hat_right_1 = msg_uhat[:, :, stage_ind, cw_ind_right[-1] + 1 :]

        u_hat = torch.cat([u_hat_left_0, u_hat_left, u_hat_right_up, u_hat_right_1], dim=2)

        msg_uhat1 = msg_uhat[:, :, :stage_ind, :]
        msg_uhat2 = msg_uhat[:, :, stage_ind + 1 :, :]
        u_hat = u_hat.unsqueeze(2)

        msg_uhat = torch.cat([msg_uhat1, u_hat, msg_uhat2], dim=2)

        return msg_uhat

    def _polar_decode_scl_pt(
        self,
        cw_ind: np.ndarray,
        msg_uhat: torch.Tensor,
        msg_llr: torch.Tensor,
        msg_pm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recursive decoding function for SCL decoding in PyTorch.

        We follow the terminology from :cite:p:`Hashemi_SSCL` and
        :cite:p:`Stimming_LLR` and branch the messages into a `left` and `right`
        update paths until reaching a leaf node.

        Tree pruning as proposed in :cite:p:`Hashemi_SSCL` is used to minimize
        the tree depth while maintaining the same output.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        if n > 1:
            # Prune tree if rate-0 subcode is detected
            if self._use_fast_scl:
                if np.sum(self._frozen_ind[cw_ind]) == n:
                    msg_pm, msg_uhat, msg_llr = self._update_rate0_code_pt(
                        msg_pm, msg_uhat, msg_llr, cw_ind
                    )
                    return msg_uhat, msg_llr, msg_pm

                if (
                    self._frozen_ind[cw_ind[-1]] == 0
                    and np.sum(self._frozen_ind[cw_ind[:-1]]) == n - 1
                ):
                    msg_pm, msg_uhat, msg_llr = self._update_rep_code_pt(
                        msg_pm, msg_uhat, msg_llr, cw_ind
                    )
                    return msg_uhat, msg_llr, msg_pm

            # Split index into left and right part
            cw_ind_left = cw_ind[: n // 2]
            cw_ind_right = cw_ind[n // 2 :]

            # ----- Left branch -----
            msg_llr = self._update_left_branch_pt(
                msg_llr, stage_ind, cw_ind_left, cw_ind_right
            )

            # Call sub-graph decoder of left branch
            msg_uhat, msg_llr, msg_pm = self._polar_decode_scl_pt(
                cw_ind_left, msg_uhat, msg_llr, msg_pm
            )

            # ----- Right branch -----
            msg_llr = self._update_right_branch_pt(
                msg_llr, msg_uhat, stage_ind, cw_ind_left, cw_ind_right
            )

            # Call sub-graph decoder of right branch
            msg_uhat, msg_llr, msg_pm = self._polar_decode_scl_pt(
                cw_ind_right, msg_uhat, msg_llr, msg_pm
            )

            # Update uhat at current stage
            msg_uhat = self._update_branch_u_pt(
                msg_uhat, stage_ind, cw_ind_left, cw_ind_right
            )

        else:
            # Leaf node: perform basic decoding op (=decision)
            msg_uhat = self._update_single_bit_pt(cw_ind, msg_uhat)
            msg_pm = self._update_pm_pt(cw_ind, msg_uhat, msg_llr, msg_pm)

            if self._frozen_ind[cw_ind] == 0:  # Position is non-frozen
                msg_pm, msg_uhat, msg_llr = self._sort_decoders_pt(
                    msg_pm, msg_uhat, msg_llr
                )
                msg_uhat, msg_llr, msg_pm = self._duplicate_paths_pt(
                    msg_uhat, msg_llr, msg_pm
                )

        return msg_uhat, msg_llr, msg_pm

    @torch.compiler.disable
    def _decode_pt_hybrid(
        self, llr_ch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hybrid SC/SCL decoding in PyTorch.

        Runs SC decoding first, checks CRC, then runs SCL only for samples
        with failed CRC. This is more efficient than full SCL when most
        samples decode correctly.

        Note:
            This function is marked with @torch.compiler.disable because
            it uses data-dependent conditional logic that causes graph breaks.
        """
        batch_size = llr_ch.shape[0]
        device = llr_ch.device
        dtype = llr_ch.dtype

        # Step 1: Run SC decoding on all samples
        u_hat_sc = self._decoder_sc(-llr_ch)

        # Step 2: Check CRC to find which samples need SCL
        # Apply input bit interleaver inverse before CRC check if needed
        if self._iil:
            u_hat_sc_crc = u_hat_sc[:, self._ind_iil_inv_t]
        else:
            u_hat_sc_crc = u_hat_sc
        _, crc_valid = self._crc_decoder(u_hat_sc_crc)
        crc_valid = crc_valid.squeeze(-1)  # [batch_size]

        # Step 3: Initialize output with SC results
        msg_uhat = torch.zeros(
            batch_size, 2 * self._list_size, 1, self._n,
            dtype=dtype, device=device
        )
        msg_pm = torch.ones(
            batch_size, 2 * self._list_size,
            dtype=dtype, device=device
        ) * self._llr_max * self.k
        msg_pm[:, 0] = 0  # SC result has zero path metric

        # Place SC results in first decoder slot at info positions
        msg_uhat[:, 0, 0, self._info_pos_t] = u_hat_sc

        # Step 4: Find samples with invalid CRC
        invalid_mask = ~crc_valid
        invalid_indices = torch.nonzero(invalid_mask, as_tuple=True)[0]

        # Step 5: Run SCL only on invalid samples (if any)
        if invalid_indices.numel() > 0:
            llr_invalid = llr_ch[invalid_indices]
            msg_uhat_scl, msg_pm_scl = self._decode_pt(llr_invalid)

            # Merge SCL results into output
            # msg_uhat_scl has shape [num_invalid, 2*L, stages+1, n]
            # We only need the final stage (index 0 after sorting)
            msg_uhat[invalid_indices] = msg_uhat_scl[:, :, 0:1, :]
            msg_pm[invalid_indices] = msg_pm_scl

        return msg_uhat, msg_pm

    @torch.compiler.disable
    def _decode_pt(
        self, llr_ch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main decoding function in PyTorch.

        Initializes memory and calls recursive decoding function.

        Note:
            This function is marked with @torch.compiler.disable because the
            recursive SCL algorithm uses position-dependent slicing that causes
            graph breaks. Disabling compilation here allows the rest of the
            model to be compiled while this runs in eager mode.
        """
        batch_size = llr_ch.shape[0]
        device = llr_ch.device
        dtype = llr_ch.dtype

        # Allocate memory for all 2*list_size decoders
        msg_uhat = torch.zeros(
            batch_size, 2 * self._list_size, self._n_stages + 1, self._n,
            dtype=dtype, device=device
        )
        msg_llr = torch.zeros(
            batch_size, 2 * self._list_size, self._n_stages, self._n,
            dtype=dtype, device=device
        )

        # Init all 2*L decoders with same llr_ch
        llr_ch_expanded = llr_ch.reshape(-1, 1, 1, self._n)
        llr_ch_expanded = llr_ch_expanded.expand(-1, 2 * self._list_size, 1, -1)

        # Init last stage with llr_ch
        msg_llr = torch.cat([msg_llr, llr_ch_expanded], dim=2)

        # Init all remaining L-1 decoders with high penalty
        pm0 = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        pm1 = self._llr_max * torch.ones(
            batch_size, self._list_size - 1, dtype=dtype, device=device
        )
        msg_pm = torch.cat([pm0, pm1, pm0, pm1], dim=1)

        # Call recursive graph function
        msg_uhat, msg_llr, msg_pm = self._polar_decode_scl_pt(
            self._cw_ind, msg_uhat, msg_llr, msg_pm
        )

        # Sort output
        msg_pm, msg_uhat, msg_llr = self._sort_decoders_pt(
            msg_pm, msg_uhat, msg_llr
        )

        return msg_uhat, msg_pm

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build and check if shape of input is invalid."""
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(
        self, llr_ch: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Successive cancellation list (SCL) decoding function.

        This function performs successive cancellation list decoding
        and returns the estimated information bits.

        An outer CRC can be applied optionally by setting ``crc_degree``.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel LLR values (as logits).

        :output b_hat: Tensor of shape `[..., k]` containing hard-decided
            estimations of all ``k`` information bits.

        :output crc_status: CRC status. Returned only if
            ``return_crc_status`` is `True`.

        Note: This function recursively unrolls the SCL decoding tree,
        thus, for larger values of ``n`` building the decoding graph can
        become time consuming. Please consider the ``cpu_only`` option
        instead.
        """
        input_shape = llr_ch.shape
        new_shape = (-1, self._n)
        llr_ch = llr_ch.reshape(new_shape)

        llr_ch = -1.0 * llr_ch  # Logits to LLRs

        # Choose decoder implementation
        if self._use_hybrid_sc:
            # Hybrid SC/SCL: use SC first, then SCL for failed CRC
            # Uses PyTorch implementation for GPU acceleration
            msg_uhat, msg_pm = self._decode_pt_hybrid(llr_ch)
        elif self._cpu_only:
            # CPU-only mode: use NumPy decoder (more memory efficient)
            llr_np = llr_ch.cpu().numpy()
            msg_uhat, msg_pm = self._decode_np_batch(llr_np)
            msg_uhat = torch.tensor(
                msg_uhat, dtype=self.dtype, device=self.device
            )
            msg_pm = torch.tensor(msg_pm, dtype=self.dtype, device=self.device)
        else:
            # Default: use PyTorch tensor-based decoder (GPU-accelerated)
            msg_uhat, msg_pm = self._decode_pt(llr_ch)

        # Check CRC (and remove CRC parity bits)
        if self._use_crc:
            # Use pre-registered tensor buffer instead of torch.tensor()
            u_hat_list = msg_uhat[:, :, 0, self._info_pos_t]

            if self._iil:
                # Use pre-registered tensor buffer
                u_hat_list_crc = u_hat_list[:, :, self._ind_iil_inv_t]
            else:
                u_hat_list_crc = u_hat_list

            _, crc_valid = self._crc_decoder(u_hat_list_crc)
            pm_penalty = (
                (1.0 - crc_valid.float()) * self._llr_max * self.k
            )
            msg_pm = msg_pm + pm_penalty.squeeze(-1)

        # Select most likely candidate
        cand_ind = torch.argmin(msg_pm, dim=-1)
        batch_indices = torch.arange(msg_uhat.shape[0], device=msg_uhat.device)
        c_hat = msg_uhat[batch_indices, cand_ind, 0, :]
        # Use pre-registered tensor buffer
        u_hat = c_hat[:, self._info_pos_t]

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self.k]
        u_hat_reshape = u_hat.reshape(output_shape)

        if self._return_crc_status:
            crc_status = crc_valid[batch_indices, cand_ind]
            output_shape_crc = list(input_shape[:-1])
            crc_status = crc_status.reshape(output_shape_crc)
            return u_hat_reshape, crc_status
        else:
            return u_hat_reshape


class PolarBPDecoder(Block):
    # pylint: disable=line-too-long
    """Belief propagation (BP) decoder for Polar codes :cite:p:`Arikan_Polar` and
    Polar-like codes based on :cite:p:`Arikan_BP` and :cite:p:`Forney_Graphs`.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param num_iter: Defining the number of decoder iterations (no early
        stopping used at the moment).
    :param hard_out: If `True`, the decoder provides hard-decided
        information bits instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :output u_hat: [..., k], `torch.float`.
        Tensor containing bit-wise soft-estimates (or hard-decided
        bit-values) of all ``k`` information bits.

    .. rubric:: Notes

    This decoder is fully differentiable and, thus, well-suited for
    gradient descent-based learning tasks such as `learned code design`
    :cite:p:`Ebada_Design`.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarBPDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarBPDecoder(frozen_pos, n, num_iter=20)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        num_iter: int = 20,
        hard_out: bool = True,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be boolean.")

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            raise ArithmeticError("Internal error: invalid info_pos generated.")

        # Register info_pos as buffer for torch.compile compatibility
        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )

        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be integer.")
        if num_iter <= 0:
            raise ValueError("num_iter must be a positive value.")
        self._num_iter = num_iter

        self._llr_max = 19.3
        self._hard_out = hard_out

        self._n_stages = int(np.log2(self._n))

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self) -> np.ndarray:
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self) -> np.ndarray:
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def num_iter(self) -> int:
        """Number of decoding iterations."""
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter: int) -> None:
        """Number of decoding iterations."""
        if not isinstance(num_iter, int):
            raise ValueError("num_iter must be int.")
        if num_iter < 0:
            raise ValueError("num_iter cannot be negative.")
        self._num_iter = num_iter

    @property
    def hard_out(self) -> bool:
        """Indicates if decoder hard-decides outputs."""
        return self._hard_out

    def _boxplus(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check-node update (boxplus) for LLR inputs."""
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        llr_out = torch.log(1 + torch.exp(x_in + y_in))
        llr_out = llr_out - torch.log(torch.exp(x_in) + torch.exp(y_in))

        return llr_out

    def _decode_bp(
        self, llr_ch: torch.Tensor, num_iter: int
    ) -> torch.Tensor:
        """Iterative BP decoding function with LLR-values."""
        bs = llr_ch.shape[0]
        device = llr_ch.device

        # Store intermediate tensors in lists
        msg_l = [[None] * (self._n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self._n_stages + 1) for _ in range(num_iter)]

        # Init frozen positions with infinity
        msg_r_in = torch.zeros((bs, self._n), dtype=self.dtype, device=device)
        msg_r_in[:, self._frozen_pos] = self._llr_max

        # Perform decoding iterations
        for ind_it in range(num_iter):
            # Update left-to-right messages
            for ind_s in range(self._n_stages):
                ind_range = np.arange(int(self._n / 2))
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s

                # Load incoming l messages
                if ind_s == self._n_stages - 1:
                    l1_in = llr_ch[:, ind_1]
                    l2_in = llr_ch[:, ind_2]
                elif ind_it == 0:
                    l1_in = torch.zeros(
                        (bs, int(self._n / 2)), dtype=self.dtype, device=device
                    )
                    l2_in = torch.zeros(
                        (bs, int(self._n / 2)), dtype=self.dtype, device=device
                    )
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in = l_in[:, ind_1]
                    l2_in = l_in[:, ind_2]

                # Load incoming r messages
                if ind_s == 0:
                    r1_in = msg_r_in[:, ind_1]
                    r2_in = msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[:, ind_1]
                    r2_in = r_in[:, ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in

                # Re-concatenate output
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
                r_out = torch.cat([r1_out, r2_out], 1)
                r_out = r_out[:, ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            # Update right-to-left messages
            for ind_s in range(self._n_stages - 1, -1, -1):
                ind_range = np.arange(int(self._n / 2))
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))

                # Load messages
                if ind_s == self._n_stages - 1:
                    l1_in = llr_ch[:, ind_1]
                    l2_in = llr_ch[:, ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in = l_in[:, ind_1]
                    l2_in = l_in[:, ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[:, ind_1]
                    r2_in = msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[:, ind_1]
                    r2_in = r_in[:, ind_2]

                # Node update functions
                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in

                l_out = torch.cat([l1_out, l2_out], 1)
                l_out = l_out[:, ind_inv]
                msg_l[ind_it][ind_s] = l_out

        # Recover u_hat using pre-registered buffer
        u_hat = msg_l[num_iter - 1][0][:, self._info_pos_t]

        if self._hard_out:
            u_hat = torch.where(
                u_hat > 0,
                torch.zeros_like(u_hat),
                torch.ones_like(u_hat),
            )
        else:
            u_hat = -1.0 * u_hat  # Re-transform to logits

        return u_hat

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build and check if shape of input is invalid."""
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape")

    def call(self, llr_ch: torch.Tensor) -> torch.Tensor:
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated information bits.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel logits/llr values.

        :output u_hat: Tensor of shape `[..., k]` containing bit-wise
            soft-estimates (or hard-decided bit-values) of all ``k``
            information bits.

        Note: This function recursively unrolls the BP decoding graph,
        thus, for larger values of ``n`` or more iterations, building the
        decoding graph can become time and memory consuming.
        """
        # Reshape inputs to [-1, n]
        input_shape = llr_ch.shape
        new_shape = (-1, self._n)
        llr_ch = llr_ch.reshape(new_shape)

        llr_ch = -1.0 * llr_ch  # Logits to LLRs

        # Decode
        u_hat = self._decode_bp(llr_ch, self._num_iter)

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self.k]
        u_hat_reshape = u_hat.reshape(output_shape)
        return u_hat_reshape


class Polar5GDecoder(Block):
    # pylint: disable=line-too-long
    """Wrapper for 5G compliant decoding including rate-recovery and CRC
    removal.

    :param enc_polar: Instance of the
        :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder` used for
        encoding including rate-matching.
    :param dec_type: Defining the decoder to be used. Must be one of
        `{"SC", "SCL", "hybSCL", "BP"}`.
    :param list_size: Defining the list size iff list-decoding is used.
        Only required for ``dec_types`` `{"SCL", "hybSCL"}`.
    :param num_iter: Defining the number of BP iterations. Only required
        for ``dec_type`` `"BP"`.
    :param return_crc_status: If `True`, the decoder additionally returns
        the CRC status indicating if a codeword was (most likely) correctly
        recovered.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :output b_hat: [..., k], `torch.float`.
        Binary tensor containing hard-decided estimations of all `k`
        information bits.

    :output crc_status: [...], `torch.bool`.
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if ``return_crc_status`` is `True`.
        Note that false positives are possible.

    .. rubric:: Notes

    This block supports the uplink and downlink Polar rate-matching scheme
    without `codeword segmentation`.

    Although the decoding `list size` is not provided by 3GPP
    :cite:p:`3GPPTS38212`, the consortium has agreed on a `list size` of 8 for
    the 5G decoding reference curves :cite:p:`Bioglio_Design`.

    All list-decoders apply `CRC-aided` decoding, however, the non-list
    decoders (`"SC"` and `"BP"`) cannot materialize the CRC leading to an
    effective rate-loss.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder

        k, n = 100, 200
        encoder = Polar5GEncoder(k, n)
        decoder = Polar5GDecoder(encoder, dec_type="SCL", list_size=8)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        enc_polar: Polar5GEncoder,
        dec_type: str = "SC",
        list_size: int = 8,
        num_iter: int = 20,
        return_crc_status: bool = False,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(enc_polar, Polar5GEncoder):
            raise TypeError("enc_polar must be Polar5GEncoder.")
        if not isinstance(dec_type, str):
            raise TypeError("dec_type must be str.")

        # Store internal attributes
        self._n_target = enc_polar.n_target
        self._k_target = enc_polar.k_target
        self._n_polar = enc_polar.n_polar
        self._k_polar = enc_polar.k_polar
        self._k_crc = enc_polar.enc_crc.crc_length
        self._bil = enc_polar._channel_type == "uplink"
        self._iil = enc_polar._channel_type == "downlink"
        self._llr_max = 100
        self._enc_polar = enc_polar
        self._dec_type = dec_type

        # Initialize the de-interleaver patterns
        self._init_interleavers()

        # Initialize decoder
        if dec_type == "SC":
            print(
                "Warning: 5G Polar codes use an integrated CRC that "
                "cannot be materialized with SC decoding and, thus, "
                "causes a degraded performance. Please consider SCL "
                "decoding instead."
            )
            self._polar_dec = PolarSCDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                precision=precision,
                device=device,
            )
        elif dec_type == "SCL":
            self._polar_dec = PolarSCLDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                crc_degree=self._enc_polar.enc_crc.crc_degree,
                list_size=list_size,
                ind_iil_inv=self.ind_iil_inv,
                precision=precision,
                device=device,
            )
        elif dec_type == "hybSCL":
            self._polar_dec = PolarSCLDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                crc_degree=self._enc_polar.enc_crc.crc_degree,
                list_size=list_size,
                use_hybrid_sc=True,
                ind_iil_inv=self.ind_iil_inv,
                precision=precision,
                device=device,
            )
        elif dec_type == "BP":
            print(
                "Warning: 5G Polar codes use an integrated CRC that "
                "cannot be materialized with BP decoding and, thus, "
                "causes a degraded performance. Please consider SCL "
                "decoding instead."
            )
            if not isinstance(num_iter, int):
                raise TypeError("num_iter must be int.")
            if num_iter <= 0:
                raise ValueError("num_iter must be positive.")
            self._num_iter = num_iter
            self._polar_dec = PolarBPDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                num_iter=num_iter,
                hard_out=True,
                precision=precision,
                device=device,
            )
        else:
            raise ValueError("Unknown value for dec_type.")

        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")

        self._return_crc_status = return_crc_status
        if self._return_crc_status:
            if dec_type in ("SCL", "hybSCL"):
                self._dec_crc = self._polar_dec._crc_decoder
            else:
                self._dec_crc = CRCDecoder(
                    self._enc_polar._enc_crc,
                    precision=precision,
                    device=device,
                )

    @property
    def k_target(self) -> int:
        """Number of information bits including rate-matching."""
        return self._k_target

    @property
    def n_target(self) -> int:
        """Codeword length including rate-matching."""
        return self._n_target

    @property
    def k_polar(self) -> int:
        """Number of information bits of mother Polar code."""
        return self._k_polar

    @property
    def n_polar(self) -> int:
        """Codeword length of mother Polar code."""
        return self._n_polar

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def dec_type(self) -> str:
        """Decoder type used for decoding as str."""
        return self._dec_type

    @property
    def polar_dec(self):
        """Decoder instance used for decoding."""
        return self._polar_dec

    def _init_interleavers(self) -> None:
        """Initialize inverse interleaver patterns for rate-recovery."""
        # Channel interleaver
        ind_ch_int = self._enc_polar.channel_interleaver(
            np.arange(self._n_target)
        )
        self.ind_ch_int_inv = np.argsort(ind_ch_int)

        # Sub-block interleaver
        ind_sub_int = self._enc_polar.subblock_interleaving(
            np.arange(self._n_polar)
        )
        self.ind_sub_int_inv = np.argsort(ind_sub_int)

        # Input bit interleaver
        if self._iil:
            self.ind_iil_inv = np.argsort(
                self._enc_polar.input_interleaver(np.arange(self._k_polar))
            )
        else:
            self.ind_iil_inv = None

        # Register as buffers for torch.compile compatibility
        self.register_buffer(
            "_ind_ch_int_inv_t",
            torch.tensor(
                self.ind_ch_int_inv, dtype=torch.int32, device=self.device
            ),
        )
        self.register_buffer(
            "_ind_sub_int_inv_t",
            torch.tensor(
                self.ind_sub_int_inv, dtype=torch.int32, device=self.device
            ),
        )
        if self._iil:
            self.register_buffer(
                "_ind_iil_inv_t",
                torch.tensor(
                    self.ind_iil_inv, dtype=torch.int32, device=self.device
                ),
            )
        else:
            self._ind_iil_inv_t = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build and check if shape of input is invalid."""
        if input_shape[-1] != self._n_target:
            raise ValueError("Invalid input shape.")

    def call(
        self, llr_ch: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Polar decoding and rate-recovery for uplink 5G Polar codes.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel logits/llr values.

        :output b_hat: Tensor of shape `[..., k]` containing hard-decided
            estimates of all ``k`` information bits.

        :output crc_status: CRC status. Returned only if
            ``return_crc_status`` is `True`.
        """
        input_shape = llr_ch.shape
        new_shape = (-1, self._n_target)
        llr_ch = llr_ch.reshape(new_shape)

        # 1.) Undo channel interleaving
        if self._bil:
            llr_deint = llr_ch[:, self._ind_ch_int_inv_t]
        else:
            llr_deint = llr_ch

        # 2.) Remove puncturing, shortening, repetition
        if self._n_target >= self._n_polar:
            # Repetition coding
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:, :n_rep]
            llr_2 = llr_deint[:, n_rep : self._n_polar]
            llr_3 = llr_deint[:, self._n_polar :]
            llr_dematched = torch.cat([llr_1 + llr_3, llr_2], 1)
        else:
            if self._k_polar / self._n_target <= 7 / 16:
                # Puncturing
                llr_zero = torch.zeros(
                    (llr_deint.shape[0], self._n_polar - self._n_target),
                    dtype=self.dtype,
                    device=llr_deint.device,
                )
                llr_dematched = torch.cat([llr_zero, llr_deint], 1)
            else:
                # Shortening
                llr_infty = (
                    -self._llr_max
                    * torch.ones(
                        (llr_deint.shape[0], self._n_polar - self._n_target),
                        dtype=self.dtype,
                        device=llr_deint.device,
                    )
                )
                llr_dematched = torch.cat([llr_deint, llr_infty], 1)

        # 3.) Remove subblock interleaving
        llr_dec = llr_dematched[:, self._ind_sub_int_inv_t]

        # 4.) Run main decoder
        u_hat_crc = self._polar_dec(llr_dec)

        # 5.) Remove input bit interleaving for downlink channels only
        if self._ind_iil_inv_t is not None:
            u_hat_crc = u_hat_crc[:, self._ind_iil_inv_t]

        # 6.) Evaluate or remove CRC (and PC)
        if self._return_crc_status:
            u_hat, crc_status = self._dec_crc(u_hat_crc)
        else:
            u_hat = u_hat_crc[:, : -self._k_crc]

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self._k_target]
        u_hat_reshape = u_hat.reshape(output_shape)
        u_hat_reshape = u_hat_reshape.to(self.dtype)

        if self._return_crc_status:
            output_shape_crc = list(input_shape[:-1])
            crc_status = crc_status.reshape(output_shape_crc)
            return u_hat_reshape, crc_status
        else:
            return u_hat_reshape

