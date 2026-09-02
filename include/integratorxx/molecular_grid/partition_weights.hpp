#pragma once

// The Becke/SSF/LKO partition-weighting math in
// impl/partition_weights.hpp is adapted from GauXC's host reference
// implementation (src/xc_integrator/local_work_driver/host/reference/weights.cxx),
// GauXC Copyright (c) 2020-2024, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy). All rights reserved.
// See GauXC's LICENSE.txt for details. This is a serial-CPU reference port
// for IntegratorXX's own direct consumers; it does not replace or mirror
// GauXC's own (in particular, device-accelerated SSF) weighting machinery.

namespace IntegratorXX {

template <typename T> class MolecularGrid;

/// Fuzzy-cell molecular partition-weighting scheme.
enum class PartitionScheme {
  Becke,  ///< Becke's original fuzzy-Voronoi partition
  SSF,    ///< Stratmann-Scuseria-Frisch partition
  LKO     ///< Laqua-Kussmann-Ochsenfeld near-field-pruned partition
};

/// Becke/SSF switching-function transition parameter (dimensionless).
constexpr double ssf_magic_factor = 0.64;
/// SSF pairwise-contribution early-exit tolerance.
constexpr double ssf_weight_tol = 1e-10;
/// LKO near-field pruning radius (bohr).
constexpr double lko_r_cutoff = 5.0;

/// @brief Apply Becke's fuzzy-Voronoi partition weights to `mg`'s
///        materialized point weights, in place.
/// @param mg The MolecularGrid to reweight; must already have its flat
///           point/weight arrays materialized by the caller.
template <typename T>
void reference_becke_partition_weights( MolecularGrid<T>& mg );

/// @brief Apply Stratmann-Scuseria-Frisch (SSF) partition weights to `mg`'s
///        materialized point weights, in place. CPU reference only -- does
///        not use or replace GauXC's own device-accelerated SSF kernels.
/// @param mg The MolecularGrid to reweight; must already have its flat
///           point/weight arrays materialized by the caller.
template <typename T>
void reference_ssf_partition_weights( MolecularGrid<T>& mg );

/// @brief Apply Laqua-Kussmann-Ochsenfeld (LKO) near-field-pruned partition
///        weights to `mg`'s materialized point weights, in place.
/// @param mg The MolecularGrid to reweight; must already have its flat
///           point/weight arrays materialized by the caller.
template <typename T>
void reference_lko_partition_weights( MolecularGrid<T>& mg );

}
