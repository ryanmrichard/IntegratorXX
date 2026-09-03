#pragma once

// Adapted from GauXC's host reference implementation
// (src/xc_integrator/local_work_driver/host/reference/weights.cxx),
// GauXC Copyright (c) 2020-2024, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy). All rights reserved.
// See GauXC's LICENSE.txt for details. This is a serial-CPU reference port
// for IntegratorXX's own direct consumers -- it does not use or replace
// GauXC's own weighting machinery (in particular, GauXC's SSF scheme has an
// actively-used, hand-tuned production GPU kernel that stays entirely in
// GauXC).
//
// All three schemes here mutate MolecularGrid::weights_cache_ in place,
// reading MolecularGrid::points_cache_/atoms_/atom_point_offset_/atom_rab_/
// atom_dist_nearest_ via friendship. Unlike GauXC's XCTask-based reference,
// no up-front sort is needed to make points atom-contiguous -- a
// MolecularGrid's flat storage already is, by atom_point_offset_.

#include <integratorxx/molecular_grid/molecular_grid.hpp>
#include <integratorxx/molecular_grid/partition_weights.hpp>
#include <integratorxx/util/fp_traits.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace IntegratorXX {

template <typename T>
void reference_becke_partition_weights( MolecularGrid<T>& mg ) {

  using traits = fp_traits<T>;
  const auto c1_5 = traits::from_real(IXX_REAL(1.5));
  const auto c0_5 = traits::from_real(IXX_REAL(0.5));
  const auto c1   = traits::from_integer(1);

  // hBecke is provably range-preserving on [-1,1] for the true real-valued
  // function (monotone, h(+-1) = +-1), so gBecke -- its 3-fold self-
  // composition -- also maps [-1,1] onto [-1,1]. Naive interval arithmetic
  // can't recognize that x is the same value on both sides of "x*x*x", so
  // each application can round outward past that provably-exact bound; the
  // clamp on every hBecke call (not just the final gBecke result) re-tightens
  // at each of the three nesting levels, since otherwise the first two
  // over-wide evaluations would poison the third.
  auto hBecke = [&]( auto x ) { return traits::clamp(c1_5*x - c0_5*x*x*x, -c1, c1); }; // Eq. 19
  auto gBecke = [&]( auto x ) { return hBecke(hBecke(hBecke(x))); };                    // Eq. 20 f_3

  const size_t natoms = mg.natoms();
  const auto&  RAB     = mg.atom_rab_;
  const auto&  points  = mg.points_cache_;
        auto&  weights = mg.weights_cache_;

  std::vector<T> partitionScratch( natoms );
  std::vector<T> atomDist( natoms );

  for( size_t ia = 0; ia < natoms; ++ia ) {
    const size_t pt_begin = mg.atom_point_offset_[ia];
    const size_t pt_end   = mg.atom_point_offset_[ia + 1];

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      const auto& point = points[ip];
      auto&       weight = weights[ip];

      // Compute distances of each center to point
      for( size_t iA = 0; iA < natoms; ++iA ) {
        const T dx = point[0] - mg.atoms_[iA].center[0];
        const T dy = point[1] - mg.atoms_[iA].center[1];
        const T dz = point[2] - mg.atoms_[iA].center[2];
        atomDist[iA] = traits::sqrt(dx*dx + dy*dy + dz*dz);
      }

      // Evaluate unnormalized partition functions
      std::fill( partitionScratch.begin(), partitionScratch.end(), c1 );
      for( size_t iA = 0; iA < natoms; ++iA )
      for( size_t jA = 0; jA < iA;     ++jA ) {
        const T mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];
        const T g  = gBecke(mu);
        partitionScratch[iA] *= c0_5 * (c1 - g);
        partitionScratch[jA] *= c0_5 * (c1 + g);
      }

      // Normalization
      T sum = traits::from_integer(0);
      for( size_t iA = 0; iA < natoms; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[ia] / sum;
    }
  }
}

template <typename T>
void reference_ssf_partition_weights( MolecularGrid<T>& mg ) {

  using traits = fp_traits<T>;
  // IXX_REAL stringifies its argument in string-literal mode, so the
  // dimensionless constants below must be spelled as literals here rather
  // than referencing ssf_magic_factor/ssf_weight_tol directly -- kept in
  // sync with the `constexpr double` definitions in partition_weights.hpp.
  static_assert(ssf_magic_factor == 0.64 && ssf_weight_tol == 1e-10,
    "keep the IXX_REAL literals below in sync with these constants");
  const auto magic  = traits::from_real(IXX_REAL(0.64));
  const auto tol    = traits::from_real(IXX_REAL(1e-10));
  const auto c1     = traits::from_integer(1);
  const auto c0_5   = traits::from_real(IXX_REAL(0.5));
  const auto c35    = traits::from_real(IXX_REAL(35.0));
  const auto c21    = traits::from_real(IXX_REAL(21.0));
  const auto c5     = traits::from_real(IXX_REAL(5.0));
  const auto c16    = traits::from_real(IXX_REAL(16.0));

  // Same self-correlation risk as gBecke (see reference_becke_partition_weights):
  // s_x is reused to build s_x2/s_x3/s_x5/s_x7, which naive interval
  // arithmetic evaluates without tracking that correlation, so the result
  // can round outward past the provably-exact [-1,1] range. Only one
  // evaluation here (no nesting), so a single clamp on the final result
  // suffices.
  auto gFrisch = [&]( auto x ) {
    const auto s_x  = x / magic;
    const auto s_x2 = s_x  * s_x;
    const auto s_x3 = s_x  * s_x2;
    const auto s_x5 = s_x3 * s_x2;
    const auto s_x7 = s_x5 * s_x2;
    return traits::clamp((c35*(s_x - s_x3) + c21*s_x5 - c5*s_x7) / c16, -c1, c1);
  };

  const size_t natoms = mg.natoms();
  const auto&  RAB          = mg.atom_rab_;
  const auto&  dist_nearest = mg.atom_dist_nearest_;
  const auto&  points       = mg.points_cache_;
        auto&  weights      = mg.weights_cache_;

  std::vector<T> partitionScratch( natoms );
  std::vector<T> atomDist( natoms );

  for( size_t ia = 0; ia < natoms; ++ia ) {
    const size_t pt_begin = mg.atom_point_offset_[ia];
    const size_t pt_end   = mg.atom_point_offset_[ia + 1];
    const T dist_cutoff = c0_5 * (c1 - magic) * dist_nearest[ia];

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      const auto& point = points[ip];
      auto&       weight = weights[ip];

      // Compute dist to parent atom
      {
        const T dx = point[0] - mg.atoms_[ia].center[0];
        const T dy = point[1] - mg.atoms_[ia].center[1];
        const T dz = point[2] - mg.atoms_[ia].center[2];
        atomDist[ia] = traits::sqrt(dx*dx + dy*dy + dz*dz);
      }

      if( atomDist[ia] < dist_cutoff ) continue; // Partition weight = 1

      // Compute distances of each (other) center to point
      for( size_t iA = 0; iA < natoms; ++iA ) {
        if( iA == ia ) continue;
        const T dx = point[0] - mg.atoms_[iA].center[0];
        const T dy = point[1] - mg.atoms_[iA].center[1];
        const T dz = point[2] - mg.atoms_[iA].center[2];
        atomDist[iA] = traits::sqrt(dx*dx + dy*dy + dz*dz);
      }

      // Evaluate unnormalized partition functions
      std::fill( partitionScratch.begin(), partitionScratch.end(), c1 );
      for( size_t iA = 0; iA < natoms; ++iA )
      for( size_t jA = 0; jA < iA;     ++jA )
      if( partitionScratch[iA] > tol ||
          partitionScratch[jA] > tol ) {

        const T mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

        if( mu <= -magic ) {
          partitionScratch[jA] = traits::from_integer(0);
        } else if( mu >= magic ) {
          partitionScratch[iA] = traits::from_integer(0);
        } else {
          const T g = c0_5 * (c1 - gFrisch(mu));
          partitionScratch[iA] *= g;
          partitionScratch[jA] *= c1 - g;
        }
      }

      // Normalization
      T sum = traits::from_integer(0);
      for( size_t iA = 0; iA < natoms; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[ia] / sum;
    }
  }
}

template <typename T>
void reference_lko_partition_weights( MolecularGrid<T>& mg ) {

  using traits = fp_traits<T>;
  const auto c1_5 = traits::from_real(IXX_REAL(1.5));
  const auto c0_5 = traits::from_real(IXX_REAL(0.5));
  const auto c1   = traits::from_integer(1);
  // See the note in reference_ssf_partition_weights: IXX_REAL stringifies
  // its argument, so this must be a literal kept in sync with lko_r_cutoff.
  static_assert(lko_r_cutoff == 5.0,
    "keep the IXX_REAL literal below in sync with lko_r_cutoff");
  const auto r_cutoff = traits::from_real(IXX_REAL(5.0));

  // See reference_becke_partition_weights for why the clamp is on every
  // hBecke call, not just the final gBecke result.
  auto hBecke = [&]( auto x ) { return traits::clamp(c1_5*x - c0_5*x*x*x, -c1, c1); };
  auto gBecke = [&]( auto x ) { return hBecke(hBecke(hBecke(x))); };

  const size_t natoms = mg.natoms();
  const auto&  RAB     = mg.atom_rab_;
  const auto&  points  = mg.points_cache_;
        auto&  weights = mg.weights_cache_;

  std::vector<T> partitionScratch( natoms );
  std::vector<T> atomDist( natoms );
  std::vector<size_t> inter_atom_dist_idx( natoms );
  std::vector<size_t> point_dist_idx( natoms );

  for( size_t iAtom = 0; iAtom < natoms; ++iAtom ) {
    const size_t pt_begin = mg.atom_point_offset_[iAtom];
    const size_t pt_end   = mg.atom_point_offset_[iAtom + 1];

    const T* RAB_parent = RAB.data() + iAtom*natoms;

    std::iota( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(), 0 );
    std::sort( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(),
      [&]( auto i, auto j ){ return RAB_parent[i] < RAB_parent[j]; } );

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      auto& weight = weights[ip];
      const auto point = points[ip];

      std::fill( atomDist.begin(), atomDist.end(), traits::infinity() );

      // Parent distance
      {
        const T dx = point[0] - mg.atoms_[iAtom].center[0];
        const T dy = point[1] - mg.atoms_[iAtom].center[1];
        const T dz = point[2] - mg.atoms_[iAtom].center[2];
        atomDist[iAtom] = traits::sqrt(dx*dx + dy*dy + dz*dz);
      }

      T r_parent  = atomDist[iAtom];
      T r_nearest = r_parent;
      size_t natoms_keep = 1;

      // Compute distances of nearby centers to point (near-field pruning)
      for( size_t iA = 1; iA < natoms; ++iA ) {
        auto idx = inter_atom_dist_idx[iA];
        if( RAB_parent[idx] > (r_parent + r_nearest + traits::from_integer(2)*r_cutoff) ) break;

        const T dx = point[0] - mg.atoms_[idx].center[0];
        const T dy = point[1] - mg.atoms_[idx].center[1];
        const T dz = point[2] - mg.atoms_[idx].center[2];
        const T r = traits::sqrt(dx*dx + dy*dy + dz*dz);

        r_nearest = std::min( r_nearest, r );
        atomDist[idx] = r;
        ++natoms_keep;
      }

      // Partition weight is 0
      if( r_parent > r_nearest + r_cutoff ) {
        weight = traits::from_integer(0);
        continue;
      }

      // Partition atom indices into a petite list of non-negligible centers
      std::iota( point_dist_idx.begin(), point_dist_idx.end(), 0 );
      auto atom_keep_end = std::partition( point_dist_idx.begin(), point_dist_idx.end(),
        [&]( auto i ){ return atomDist[i] < traits::infinity(); } );

      // Only sort over non-negligible centers
      std::sort( point_dist_idx.begin(), atom_keep_end,
        [&]( auto i, auto j ){ return atomDist[i] < atomDist[j]; } );

      // Get parent index within the petite list
      auto parent_it  = std::find( point_dist_idx.begin(), atom_keep_end, iAtom );
      auto parent_idx = static_cast<size_t>( std::distance( point_dist_idx.begin(), parent_it ) );

      // Sort atom distances for contiguous reads in the weight loop
      auto atom_dist_end = std::partition( atomDist.begin(), atomDist.end(),
        [&]( auto x ){ return x < traits::infinity(); } );
      std::sort( atomDist.begin(), atom_dist_end );

      // Evaluate unnormalized partition functions
      std::fill_n( partitionScratch.begin(), natoms_keep, traits::from_integer(0) );
      for( size_t i = 0; i < natoms_keep; ++i ) {
        auto   idx_i = point_dist_idx[i];
        auto   r_i   = atomDist[i];
        if( r_i > (r_nearest + r_cutoff) ) break;
        partitionScratch[i] = c1;

        const T* RAB_i = RAB.data() + idx_i*natoms;

        for( size_t j = 0; j < i; ++j ) {
          auto r_j = atomDist[j];
          if( r_j > (r_i + r_cutoff) ) break;

          auto idx_j = point_dist_idx[j];
          const T mu = (r_i - r_j) / std::min( RAB_i[idx_j], r_cutoff );

          const T g     = gBecke(mu);
          const T s_ij  = c0_5 * (c1 - g);
          partitionScratch[i] *= s_ij;
          partitionScratch[j] *= c1 - s_ij;
        }
      }

      // Normalization
      T sum = traits::from_integer(0);
      for( size_t iA = 0; iA < natoms_keep; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[parent_idx] / sum;
    }
  }
}

}
