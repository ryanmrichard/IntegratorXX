#pragma once

// The per-element radius/scaling-factor tables and grid-size presets in
// impl/defaults.hpp are adapted from GauXC's MolGridFactory
// (include/gauxc/molgrid/defaults.hpp, src/molgrid_defaults.cxx,
// src/atomic_radii.cxx), GauXC Copyright (c) 2020-2024, The Regents of the
// University of California, through Lawrence Berkeley National Laboratory
// (subject to receipt of any required approvals from the U.S. Dept. of
// Energy). All rights reserved. See GauXC's LICENSE.txt for details.

#include <integratorxx/molecular_grid/molecular_grid.hpp>
#include <integratorxx/generators/spherical_factory.hpp>

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace IntegratorXX {

/// Plain atomic-number alias -- deliberately not a Molecule/chemist type,
/// so this header stays free of any dependency on a specific molecule
/// representation.
using AtomicId = int64_t;

/// Standard atomic-grid size presets.
enum class AtomicGridSizeDefault { FineGrid, UltraFineGrid, SuperFineGrid, GM3, GM5 };

/// Slater, J.C. J. Chem. Phys. 41, 3199, 1964. https://doi.org/10.1063/1.1725697
double slater_radius_64( AtomicId );
/// Slater, J.C. Phys. Rev. 36, 57, 1930. https://doi.org/10.1103/PhysRev.36.57
double slater_radius_30( AtomicId );
/// Clementi, E., Raimondi, D.L., Reinhardt, W.P. J. Chem. Phys. 47, 1300, 1967.
/// https://doi.org/10.1063/1.1712084
double clementi_radius_67( AtomicId );
/// Best available tabulated atomic radius for `Z` (Slater-64, falling back
/// to Clementi-67, falling back to a fixed default).
double default_atomic_radius( AtomicId );

double default_mk_radial_scaling_factor( AtomicId );
double default_mhl_radial_scaling_factor( AtomicId );
double default_ta_radial_scaling_factor( AtomicId );
double default_radial_scaling_factor( RadialQuad, AtomicId );

/// @returns (radial_size, angular_size) for a standard grid preset.
std::tuple<size_t,size_t> default_grid_size( AtomicId, RadialQuad, AtomicGridSizeDefault );

/// @brief Element-indexed default grid-specification builders, filling the
///        gap left by SphericalGridFactory (which has no per-element
///        presets of its own).
struct MolecularGridDefaults {

  static UnprunedSphericalGridSpecification create_default_unpruned_grid_spec(
    AtomicId, RadialQuad, size_t radial_size, size_t angular_size );

  static UnprunedSphericalGridSpecification create_default_unpruned_grid_spec(
    AtomicId, RadialQuad, AtomicGridSizeDefault );

  template <typename... Args>
  static PrunedSphericalGridSpecification create_default_pruned_grid_spec(
    PruningScheme scheme, Args&&... args ) {
    return create_pruned_spec( scheme,
      create_default_unpruned_grid_spec(std::forward<Args>(args)...) );
  }

  /// @param unique_ids Unique atomic numbers to build specs for (unlike
  ///        GauXC's Molecule-taking equivalent, this only ever needs the
  ///        set of distinct elements present, not a full molecule).
  template <typename... Args>
  static std::unordered_map<AtomicId, PrunedSphericalGridSpecification>
    create_default_grid_spec_map( const std::vector<AtomicId>& unique_ids,
      PruningScheme scheme, Args&&... args ) {

    std::unordered_map<AtomicId, PrunedSphericalGridSpecification> molmap;
    for( auto id : unique_ids )
    if( !molmap.count(id) ) {
      molmap.emplace( id,
        create_default_pruned_grid_spec(scheme, id, std::forward<Args>(args)...) );
    }
    return molmap;
  }

  static std::unordered_map<AtomicId, SphericalGridFactory::spherical_grid_ptr>
    generate_gridmap(
      const std::unordered_map<AtomicId, PrunedSphericalGridSpecification>& gs_map );

};

/// @brief Glue between the "one grid template per element" world of
///        MolecularGridDefaults::generate_gridmap and MolecularGrid's
///        "one AtomInstance per atom instance" world.
/// @param atomic_ids Per-atom-instance atomic numbers, same length/order as `positions`.
/// @param positions  Per-atom-instance molecular-frame positions.
/// @param element_grids Map from atomic number to a (possibly shared) atomic-grid template.
/// @throws std::out_of_range if an id in `atomic_ids` has no entry in `element_grids`.
std::vector<AtomInstance> make_atom_instances(
  const std::vector<AtomicId>&               atomic_ids,
  const std::vector<cartesian_pt_t<double>>& positions,
  const std::unordered_map<AtomicId, SphericalGridFactory::spherical_grid_ptr>& element_grids );

}
