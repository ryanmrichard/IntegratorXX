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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace IntegratorXX {

void reference_becke_partition_weights( MolecularGrid& mg ) {

  auto hBecke = []( double x ) { return 1.5*x - 0.5*x*x*x; };       // Eq. 19
  auto gBecke = [&]( double x ) { return hBecke(hBecke(hBecke(x))); }; // Eq. 20 f_3

  const size_t natoms = mg.natoms();
  const auto&  RAB     = mg.atom_rab_;
  const auto&  points  = mg.points_cache_;
        auto&  weights = mg.weights_cache_;

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t ia = 0; ia < natoms; ++ia ) {
    const size_t pt_begin = mg.atom_point_offset_[ia];
    const size_t pt_end   = mg.atom_point_offset_[ia + 1];

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      const auto& point = points[ip];
      auto&       weight = weights[ip];

      // Compute distances of each center to point
      for( size_t iA = 0; iA < natoms; ++iA ) {
        const double dx = point[0] - mg.atoms_[iA].center[0];
        const double dy = point[1] - mg.atoms_[iA].center[1];
        const double dz = point[2] - mg.atoms_[iA].center[2];
        atomDist[iA] = std::sqrt(dx*dx + dy*dy + dz*dz);
      }

      // Evaluate unnormalized partition functions
      std::fill( partitionScratch.begin(), partitionScratch.end(), 1. );
      for( size_t iA = 0; iA < natoms; ++iA )
      for( size_t jA = 0; jA < iA;     ++jA ) {
        const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];
        const double g  = gBecke(mu);
        partitionScratch[iA] *= 0.5 * (1. - g);
        partitionScratch[jA] *= 0.5 * (1. + g);
      }

      // Normalization
      double sum = 0.;
      for( size_t iA = 0; iA < natoms; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[ia] / sum;
    }
  }
}

void reference_ssf_partition_weights( MolecularGrid& mg ) {

  auto gFrisch = []( double x ) {
    const double s_x  = x / ssf_magic_factor;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;
    return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
  };

  const size_t natoms = mg.natoms();
  const auto&  RAB          = mg.atom_rab_;
  const auto&  dist_nearest = mg.atom_dist_nearest_;
  const auto&  points       = mg.points_cache_;
        auto&  weights      = mg.weights_cache_;

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );

  for( size_t ia = 0; ia < natoms; ++ia ) {
    const size_t pt_begin = mg.atom_point_offset_[ia];
    const size_t pt_end   = mg.atom_point_offset_[ia + 1];
    const double dist_cutoff = 0.5 * (1. - ssf_magic_factor) * dist_nearest[ia];

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      const auto& point = points[ip];
      auto&       weight = weights[ip];

      // Compute dist to parent atom
      {
        const double dx = point[0] - mg.atoms_[ia].center[0];
        const double dy = point[1] - mg.atoms_[ia].center[1];
        const double dz = point[2] - mg.atoms_[ia].center[2];
        atomDist[ia] = std::sqrt(dx*dx + dy*dy + dz*dz);
      }

      if( atomDist[ia] < dist_cutoff ) continue; // Partition weight = 1

      // Compute distances of each (other) center to point
      for( size_t iA = 0; iA < natoms; ++iA ) {
        if( iA == ia ) continue;
        const double dx = point[0] - mg.atoms_[iA].center[0];
        const double dy = point[1] - mg.atoms_[iA].center[1];
        const double dz = point[2] - mg.atoms_[iA].center[2];
        atomDist[iA] = std::sqrt(dx*dx + dy*dy + dz*dz);
      }

      // Evaluate unnormalized partition functions
      std::fill( partitionScratch.begin(), partitionScratch.end(), 1. );
      for( size_t iA = 0; iA < natoms; ++iA )
      for( size_t jA = 0; jA < iA;     ++jA )
      if( partitionScratch[iA] > ssf_weight_tol ||
          partitionScratch[jA] > ssf_weight_tol ) {

        const double mu = (atomDist[iA] - atomDist[jA]) / RAB[jA + iA*natoms];

        if( mu <= -ssf_magic_factor ) {
          partitionScratch[jA] = 0.;
        } else if( mu >= ssf_magic_factor ) {
          partitionScratch[iA] = 0.;
        } else {
          const double g = 0.5 * (1. - gFrisch(mu));
          partitionScratch[iA] *= g;
          partitionScratch[jA] *= 1. - g;
        }
      }

      // Normalization
      double sum = 0.;
      for( size_t iA = 0; iA < natoms; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[ia] / sum;
    }
  }
}

void reference_lko_partition_weights( MolecularGrid& mg ) {

  auto hBecke = []( double x ) { return 1.5*x - 0.5*x*x*x; };
  auto gBecke = [&]( double x ) { return hBecke(hBecke(hBecke(x))); };

  const size_t natoms = mg.natoms();
  const auto&  RAB     = mg.atom_rab_;
  const auto&  points  = mg.points_cache_;
        auto&  weights = mg.weights_cache_;

  std::vector<double> partitionScratch( natoms );
  std::vector<double> atomDist( natoms );
  std::vector<size_t> inter_atom_dist_idx( natoms );
  std::vector<size_t> point_dist_idx( natoms );

  for( size_t iAtom = 0; iAtom < natoms; ++iAtom ) {
    const size_t pt_begin = mg.atom_point_offset_[iAtom];
    const size_t pt_end   = mg.atom_point_offset_[iAtom + 1];

    const double* RAB_parent = RAB.data() + iAtom*natoms;

    std::iota( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(), 0 );
    std::sort( inter_atom_dist_idx.begin(), inter_atom_dist_idx.end(),
      [&]( auto i, auto j ){ return RAB_parent[i] < RAB_parent[j]; } );

    for( size_t ip = pt_begin; ip < pt_end; ++ip ) {
      auto& weight = weights[ip];
      const auto point = points[ip];

      std::fill( atomDist.begin(), atomDist.end(), std::numeric_limits<double>::infinity() );

      // Parent distance
      {
        const double dx = point[0] - mg.atoms_[iAtom].center[0];
        const double dy = point[1] - mg.atoms_[iAtom].center[1];
        const double dz = point[2] - mg.atoms_[iAtom].center[2];
        atomDist[iAtom] = std::sqrt(dx*dx + dy*dy + dz*dz);
      }

      double r_parent  = atomDist[iAtom];
      double r_nearest = r_parent;
      size_t natoms_keep = 1;

      // Compute distances of nearby centers to point (near-field pruning)
      for( size_t iA = 1; iA < natoms; ++iA ) {
        auto idx = inter_atom_dist_idx[iA];
        if( RAB_parent[idx] > (r_parent + r_nearest + 2*lko_r_cutoff) ) break;

        const double dx = point[0] - mg.atoms_[idx].center[0];
        const double dy = point[1] - mg.atoms_[idx].center[1];
        const double dz = point[2] - mg.atoms_[idx].center[2];
        const double r = std::sqrt(dx*dx + dy*dy + dz*dz);

        r_nearest = std::min( r_nearest, r );
        atomDist[idx] = r;
        ++natoms_keep;
      }

      // Partition weight is 0
      if( r_parent > r_nearest + lko_r_cutoff ) {
        weight = 0.;
        continue;
      }

      // Partition atom indices into a petite list of non-negligible centers
      std::iota( point_dist_idx.begin(), point_dist_idx.end(), 0 );
      auto atom_keep_end = std::partition( point_dist_idx.begin(), point_dist_idx.end(),
        [&]( auto i ){ return atomDist[i] < std::numeric_limits<double>::infinity(); } );

      // Only sort over non-negligible centers
      std::sort( point_dist_idx.begin(), atom_keep_end,
        [&]( auto i, auto j ){ return atomDist[i] < atomDist[j]; } );

      // Get parent index within the petite list
      auto parent_it  = std::find( point_dist_idx.begin(), atom_keep_end, iAtom );
      auto parent_idx = static_cast<size_t>( std::distance( point_dist_idx.begin(), parent_it ) );

      // Sort atom distances for contiguous reads in the weight loop
      auto atom_dist_end = std::partition( atomDist.begin(), atomDist.end(),
        []( auto x ){ return x < std::numeric_limits<double>::infinity(); } );
      std::sort( atomDist.begin(), atom_dist_end );

      // Evaluate unnormalized partition functions
      std::fill_n( partitionScratch.begin(), natoms_keep, 0. );
      for( size_t i = 0; i < natoms_keep; ++i ) {
        auto   idx_i = point_dist_idx[i];
        auto   r_i   = atomDist[i];
        if( r_i > (r_nearest + lko_r_cutoff) ) break;
        partitionScratch[i] = 1.;

        const double* RAB_i = RAB.data() + idx_i*natoms;

        for( size_t j = 0; j < i; ++j ) {
          auto r_j = atomDist[j];
          if( r_j > (r_i + lko_r_cutoff) ) break;

          auto idx_j = point_dist_idx[j];
          const double mu = (r_i - r_j) / std::min( RAB_i[idx_j], lko_r_cutoff );

          const double g     = gBecke(mu);
          const double s_ij  = 0.5 * (1. - g);
          partitionScratch[i] *= s_ij;
          partitionScratch[j] *= 1. - s_ij;
        }
      }

      // Normalization
      double sum = 0.;
      for( size_t iA = 0; iA < natoms_keep; ++iA ) sum += partitionScratch[iA];

      // Update weight
      weight *= partitionScratch[parent_idx] / sum;
    }
  }
}

}
