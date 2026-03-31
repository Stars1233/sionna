#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 LSP classes"""

import numpy as np
import pytest
import torch
from scipy.stats import kstest, norm

from sionna.phy import PI, config
from sionna.phy.channel import tr38901
from sionna.phy.channel.tr38901 import LSP, LSPGenerator

from channel_test_utils import (
    channel_test_on_models,
    corr_dist_asd,
    corr_dist_asa,
    corr_dist_ds,
    corr_dist_k,
    corr_dist_sf,
    corr_dist_zsa,
    corr_dist_zsd,
    cross_corr,
    generate_random_bool,
    generate_random_loc,
    limited_normal,
    log10ASD,
    log10ASA,
    log10DS,
    log10K_dB,
    log10SF_dB,
    log10ZSA,
    log10ZSD,
    los_probability,
    pathloss,
    pathloss_std,
    zod_offset,
)


class TestLSP:
    """Tests for the LSP data class"""

    def test_lsp_instantiation(self, device, precision):
        """Test that LSP can be instantiated with the expected shapes"""
        batch_size = 10
        num_tx = 3
        num_rx = 5

        # Create random tensors with expected shapes
        ds = torch.rand(batch_size, num_tx, num_rx, device=device)
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        # Verify all attributes are correctly assigned
        assert torch.equal(lsp.ds, ds)
        assert torch.equal(lsp.asd, asd)
        assert torch.equal(lsp.asa, asa)
        assert torch.equal(lsp.sf, sf)
        assert torch.equal(lsp.k_factor, k_factor)
        assert torch.equal(lsp.zsa, zsa)
        assert torch.equal(lsp.zsd, zsd)

    def test_lsp_ds_positive(self, device, precision):
        """Test that delay spread is positive"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        # Create positive values
        ds = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 1e-9
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.ds > 0)

    def test_lsp_angle_spreads_bounded(self, device, precision):
        """Test that angle spreads are bounded (degrees)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        # Create bounded angle spread values (0 to 104 degrees for azimuth, 0 to 52 for zenith)
        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 104
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 104
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 52
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 52

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        # Check azimuth angle spreads are bounded by 104 degrees
        assert torch.all(lsp.asd <= 104)
        assert torch.all(lsp.asa <= 104)

        # Check zenith angle spreads are bounded by 52 degrees
        assert torch.all(lsp.zsa <= 52)
        assert torch.all(lsp.zsd <= 52)

    def test_lsp_k_factor_positive(self, device, precision):
        """Test that K-factor is positive (linear scale)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        # K-factor should be positive (linear scale)
        k_factor = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 0.1
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 30
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 30

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.k_factor > 0)

    def test_lsp_sf_positive(self, device, precision):
        """Test that shadow fading is positive (linear scale)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        # SF should be positive (linear scale)
        sf = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 0.1
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 30
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 30

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.sf > 0)

    def test_lsp_shapes_consistent(self, device, precision):
        """Test that all LSP attributes have consistent shapes"""
        batch_size = 8
        num_tx = 4
        num_rx = 6

        ds = torch.rand(batch_size, num_tx, num_rx, device=device)
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        expected_shape = (batch_size, num_tx, num_rx)

        assert lsp.ds.shape == expected_shape
        assert lsp.asd.shape == expected_shape
        assert lsp.asa.shape == expected_shape
        assert lsp.sf.shape == expected_shape
        assert lsp.k_factor.shape == expected_shape
        assert lsp.zsa.shape == expected_shape
        assert lsp.zsd.shape == expected_shape


class TestLSPGenerator:
    """Tests for LSPGenerator matching TensorFlow implementation tests."""

    # Test configuration
    CARRIER_FREQUENCY = 3.5e9  # Hz
    H_UT = 1.5  # Height of UTs
    H_BS = 35.0  # Height of BSs
    BATCH_SIZE = 100000  # Large batch for statistical tests
    NB_UT = 5  # Number of UTs for spatial correlation tests

    # Test thresholds
    MAX_ERR_KS = 1e-2  # Maximum KS statistic for distribution tests
    MAX_ERR_CROSS_CORR = 3e-2  # Maximum error for cross-correlation
    MAX_ERR_SPAT_CORR = 3e-2  # Maximum error for spatial correlation
    MAX_ERR_LOS_PROB = 1e-2  # Maximum error for LoS probability
    MAX_ERR_ZOD_OFFSET = 1e-2  # Maximum error for ZOD offset
    MAX_ERR_PATHLOSS_MEAN = 1.0  # Maximum error for pathloss mean
    MAX_ERR_PATHLOSS_STD = 1e-1  # Maximum error for pathloss std

    @pytest.fixture(scope="class")
    def lsp_samples(self, request):
        """Sample LSPs from all channel models for testing.
        
        Uses the device specified by --device flag.
        """
        device_option = request.config.getoption("--device", default="gpu")
        if device_option == "cpu":
            device = "cpu"
        elif device_option == "gpu" and torch.cuda.is_available():
            device = "cuda:0"
        elif device_option == "all" and torch.cuda.is_available():
            device = "cuda:0"  # Use GPU for class-scoped fixture when "all"
        else:
            device = "cpu"
        batch_size = self.BATCH_SIZE
        nb_bs = 1
        nb_ut = self.NB_UT
        fc = self.CARRIER_FREQUENCY
        h_ut = self.H_UT
        h_bs = self.H_BS
        dtype = torch.float64

        # Create antenna arrays
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )

        # Generate shared topology
        ut_orientations = torch.zeros(batch_size, nb_ut, dtype=dtype, device=device)
        bs_orientations = torch.zeros(batch_size, nb_bs, dtype=dtype, device=device)
        ut_velocities = torch.zeros(batch_size, nb_ut, dtype=dtype, device=device)

        ut_loc = generate_random_loc(
            batch_size, nb_ut, (100, 2000), (100, 2000), (h_ut, h_ut),
            share_loc=True, dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            batch_size, nb_bs, (0, 100), (0, 100), (h_bs, h_bs),
            share_loc=True, dtype=dtype, device=device
        )

        samples = {}

        # Test all scenarios
        for model_name, scenario_cls in [
            ("rma", tr38901.RMaScenario),
            ("umi", tr38901.UMiScenario),
            ("uma", tr38901.UMaScenario),
        ]:
            samples[model_name] = {}

            # Create scenario
            if model_name == "rma":
                scenario = scenario_cls(
                    fc, ut_array, bs_array, "uplink", precision="double", device=device
                )
            else:
                scenario = scenario_cls(
                    fc, "low", ut_array, bs_array, "uplink", precision="double", device=device
                )

            lsp_sampler = LSPGenerator(scenario)

            # LoS
            in_state = generate_random_bool(batch_size, nb_ut, 0.0, device=device)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state, los=True
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["los"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # NLoS
            in_state = generate_random_bool(batch_size, nb_ut, 0.0, device=device)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state, los=False
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["nlos"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # O2I
            in_state = generate_random_bool(batch_size, nb_ut, 1.0, device=device)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["o2i"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # Store scenario info — use .copy() to decouple from in-place-updated buffers
            samples[model_name]["los_prob"] = scenario.los_probability.cpu().numpy().copy()
            samples[model_name]["d_2d"] = scenario.distance_2d.cpu().numpy().copy()
            samples[model_name]["d_2d_ut"] = scenario.matrix_ut_distance_2d.cpu().numpy().copy()
            samples[model_name]["d_2d_out"] = scenario.distance_2d_out.cpu().numpy().copy()
            samples[model_name]["d_3d"] = scenario.distance_3d[0, 0, :].cpu().numpy().copy()
            if model_name == "rma":
                samples[model_name]["w"] = scenario.average_street_width
                samples[model_name]["h"] = scenario.average_building_height

        return samples

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_ds_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP DS (delay spread)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.ds[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10DS(model, submodel, self.CARRIER_FREQUENCY)
        D, _ = kstest(samples, norm.cdf, args=(mu, std))

        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} DS distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_asa_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ASA (azimuth angle spread of arrival)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.asa[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ASA(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(104), f"{model}:{submodel} ASA exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ASA distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_asd_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ASD (azimuth angle spread of departure)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.asd[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ASD(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(104), f"{model}:{submodel} ASD exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ASD distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zsa_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ZSA (zenith angle spread of arrival)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.zsa[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ZSA(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(52), f"{model}:{submodel} ZSA exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ZSA distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zsd_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ZSD (zenith angle spread of departure)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d = lsp_samples[model]["d_2d"][0, 0, 0]
        samples = lsp.zsd[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ZSD(
            model, submodel, d_2d, self.CARRIER_FREQUENCY, self.H_BS, self.H_UT
        )
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(52), f"{model}:{submodel} ZSD exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ZSD distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_sf_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP SF (shadow fading)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d = lsp_samples[model]["d_2d"][0, 0, 0]
        samples = lsp.sf[:, 0, 0].cpu().numpy()
        samples = 10.0 * np.log10(samples)  # Convert to dB

        mu, std = log10SF_dB(
            model, submodel, d_2d, self.CARRIER_FREQUENCY, self.H_BS, self.H_UT
        )
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} SF distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    def test_k_factor_distribution(self, lsp_samples, model):
        """Test the distribution of LSP K-factor (LoS only)."""
        lsp = lsp_samples[model]["los"]["lsp"]
        samples = lsp.k_factor[:, 0, 0].cpu().numpy()
        samples = 10.0 * np.log10(samples)  # Convert to dB

        mu, std = log10K_dB(model, "los")
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        assert D <= self.MAX_ERR_KS, f"{model}:los K-factor distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_cross_correlation(self, lsp_samples, model, submodel):
        """Test the LSP cross-correlation matrix."""
        lsp = lsp_samples[model][submodel]["lsp"]

        lsp_list = []
        lsp_list.append(np.log10(lsp.ds[:, 0, 0].cpu().numpy()))
        lsp_list.append(np.log10(lsp.asd[:, 0, 0].cpu().numpy()))
        lsp_list.append(np.log10(lsp.asa[:, 0, 0].cpu().numpy()))
        lsp_list.append(np.log10(lsp.sf[:, 0, 0].cpu().numpy()))
        if submodel == "los":
            lsp_list.append(np.log10(lsp.k_factor[:, 0, 0].cpu().numpy()))
        lsp_list.append(np.log10(lsp.zsa[:, 0, 0].cpu().numpy()))
        lsp_list.append(np.log10(lsp.zsd[:, 0, 0].cpu().numpy()))

        lsp_list = np.stack(lsp_list, axis=-1)
        cross_corr_measured = np.corrcoef(lsp_list.T)
        abs_err = np.abs(cross_corr(model, submodel) - cross_corr_measured)
        max_err = np.max(abs_err)

        assert max_err <= self.MAX_ERR_CROSS_CORR, (
            f"{model}:{submodel} cross-correlation failed"
        )

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_spatial_correlation(self, lsp_samples, model, submodel):
        """Test the spatial correlation of LSPs."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d_ut = lsp_samples[model]["d_2d_ut"][0, 0]

        # Get measured correlations
        ds_samples = np.log10(lsp.ds[:, 0, :].cpu().numpy())
        asd_samples = np.log10(lsp.asd[:, 0, :].cpu().numpy())
        asa_samples = np.log10(lsp.asa[:, 0, :].cpu().numpy())
        sf_samples = np.log10(lsp.sf[:, 0, :].cpu().numpy())
        if submodel == "los":
            k_samples = np.log10(lsp.k_factor[:, 0, :].cpu().numpy())
        zsa_samples = np.log10(lsp.zsa[:, 0, :].cpu().numpy())
        zsd_samples = np.log10(lsp.zsd[:, 0, :].cpu().numpy())

        C_ds_measured = np.corrcoef(ds_samples.T)[0]
        C_asd_measured = np.corrcoef(asd_samples.T)[0]
        C_asa_measured = np.corrcoef(asa_samples.T)[0]
        C_sf_measured = np.corrcoef(sf_samples.T)[0]
        if submodel == "los":
            C_k_measured = np.corrcoef(k_samples.T)[0]
        C_zsa_measured = np.corrcoef(zsa_samples.T)[0]
        C_zsd_measured = np.corrcoef(zsd_samples.T)[0]

        # Reference correlations
        C_ds = np.exp(-d_2d_ut / corr_dist_ds(model, submodel))
        C_asd = np.exp(-d_2d_ut / corr_dist_asd(model, submodel))
        C_asa = np.exp(-d_2d_ut / corr_dist_asa(model, submodel))
        C_sf = np.exp(-d_2d_ut / corr_dist_sf(model, submodel))
        if submodel == "los":
            C_k = np.exp(-d_2d_ut / corr_dist_k(model, submodel))
        C_zsa = np.exp(-d_2d_ut / corr_dist_zsa(model, submodel))
        C_zsd = np.exp(-d_2d_ut / corr_dist_zsd(model, submodel))

        # Check errors
        assert np.max(np.abs(C_ds_measured - C_ds)) <= self.MAX_ERR_SPAT_CORR
        assert np.max(np.abs(C_asd_measured - C_asd)) <= self.MAX_ERR_SPAT_CORR
        assert np.max(np.abs(C_asa_measured - C_asa)) <= self.MAX_ERR_SPAT_CORR
        assert np.max(np.abs(C_sf_measured - C_sf)) <= self.MAX_ERR_SPAT_CORR
        if submodel == "los":
            assert np.max(np.abs(C_k_measured - C_k)) <= self.MAX_ERR_SPAT_CORR
        assert np.max(np.abs(C_zsa_measured - C_zsa)) <= self.MAX_ERR_SPAT_CORR
        assert np.max(np.abs(C_zsd_measured - C_zsd)) <= self.MAX_ERR_SPAT_CORR

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    def test_los_probability(self, lsp_samples, model):
        """Test LoS probability calculation."""
        d_2d_out = lsp_samples[model]["d_2d_out"]
        los_prob_ref = los_probability(model, d_2d_out, self.H_UT)
        los_prob = lsp_samples[model]["los_prob"]
        max_err = np.max(np.abs(los_prob_ref - los_prob))

        assert max_err <= self.MAX_ERR_LOS_PROB, f"{model} LoS probability failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zod_offset(self, lsp_samples, model, submodel):
        """Test ZOD offset calculation."""
        d_2d = lsp_samples[model]["d_2d"]
        samples = lsp_samples[model][submodel]["zod_offset"]
        samples_ref = zod_offset(model, submodel, self.CARRIER_FREQUENCY, d_2d, self.H_UT)
        max_err = np.max(np.abs(samples - samples_ref))

        assert max_err <= self.MAX_ERR_ZOD_OFFSET, f"{model}:{submodel} ZOD offset failed"
