#pragma once

#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/batch/spherical_micro_batcher.hpp>
#include <integratorxx/types.hpp>
#include <integratorxx/util/fp_traits.hpp>
#include <integratorxx/molecular_grid/partition_weights.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace IntegratorXX {

/// @brief One atom instance contributing to a MolecularGrid: a
///        molecular-frame position plus the (possibly shared) atomic-grid
///        template used to generate its points.
///
/// Several AtomInstances may share the SAME `grid` (e.g. all atoms of one
/// element reuse one SphericalGridFactory-built template); MolecularGrid
/// clones it internally via SphericalQuadratureBase::clone() and never
/// mutates the shared template.
template <typename T>
struct AtomInstance {
  using spherical_grid_ptr = typename SphericalGridFactory<T>::spherical_grid_ptr;
  cartesian_pt_t<T>  center;
  spherical_grid_ptr grid;
};

/// @brief Cheap, non-materializing metadata for one molecular-grid batch.
///
/// Available for every batch without ever copying that batch's point/weight
/// vectors -- backed directly by the already-resident storage of the owning
/// atom's SphericalMicroBatcher.
template <typename T>
struct BatchInfo {
  size_t atom_index        = 0;  ///< index into the MolecularGrid's atom list
  size_t local_batch_index = 0;  ///< index into that atom's own batch list
  size_t point_begin       = 0;  ///< global flat point-index range [point_begin,point_end)
  size_t point_end         = 0;
  cartesian_pt_t<T> box_lo{};
  cartesian_pt_t<T> box_up{};
};

/// @brief One materialized (owned-copy) molecular-grid batch.
///
/// Matches SphericalMicroBatcher::at()'s value-return convention: mutating
/// a returned MolecularGridBatch does not feed back into the owning
/// MolecularGrid.
template <typename T>
struct MolecularGridBatch : BatchInfo<T> {
  std::vector<cartesian_pt_t<T>> points;
  std::vector<T>                 weights;
};

/// @brief Assembles a multi-atom integration grid from per-atom "atomic
///        grids" (as produced by SphericalGridFactory) and atomic positions.
///
/// Construction recenters a clone of each atom's atomic-grid template at
/// its molecular-frame position and micro-batches it, but does *not* copy
/// any batch's points/weights into a separate flat array -- cheap metadata
/// (point counts, batch counts, bounding boxes) is available immediately
/// via atom_npts()/atom_nbatches()/batch_info(), while the actual
/// point/weight vector materialization happens only for batches a caller
/// explicitly asks for, via batch()/batches_for_atoms()/
/// batches_for_point_range(), or for the whole molecule via the lazily
/// cached points()/weights() (also the precondition for
/// apply_partition_weights()). This is a deliberate design choice -- see
/// the class-level discussion in the IntegratorXX repository's design notes
/// for why full eager materialization at construction is avoided.
template <typename T>
class MolecularGrid {
public:
  using spherical_grid_ptr = typename SphericalGridFactory<T>::spherical_grid_ptr;
  using point_type         = cartesian_pt_t<T>;
  using point_container    = std::vector<point_type>;
  using weight_container   = std::vector<T>;
  using batcher_type       = SphericalMicroBatcher<point_container, weight_container>;

  MolecularGrid() = default;

  /// @param atoms       Per-atom-instance (position, atomic-grid template) pairs.
  /// @param max_batch_sz Maximum number of points per micro-batch, forwarded
  ///                      to each atom's SphericalMicroBatcher.
  MolecularGrid( std::vector<AtomInstance<T>> atoms, size_t max_batch_sz ) :
    atoms_( std::move(atoms) ) {

    const size_t na = atoms_.size();
    atom_batchers_.reserve(na);
    atom_point_offset_.assign(na + 1, 0);
    atom_batch_offset_.assign(na + 1, 0);

    for( size_t ia = 0; ia < na; ++ia ) {
      if( !atoms_[ia].grid )
        throw std::runtime_error("MolecularGrid: AtomInstance has a null atomic grid");

      auto cloned = atoms_[ia].grid->clone();
      cloned->recenter( atoms_[ia].center );
      atom_batchers_.emplace_back( max_batch_sz, cloned );

      atom_point_offset_[ia + 1] = atom_point_offset_[ia] + atom_batchers_[ia].npts();
      atom_batch_offset_[ia + 1] = atom_batch_offset_[ia] + atom_batchers_[ia].nbatches();
    }

    const size_t total_batches = atom_batch_offset_.back();
    batch_atom_index_.reserve(total_batches);
    batch_local_index_.reserve(total_batches);
    batch_point_offset_.reserve(total_batches + 1);

    size_t running_pt_offset = 0;
    for( size_t ia = 0; ia < na; ++ia ) {
      const size_t nb = atom_batchers_[ia].nbatches();
      for( size_t ib = 0; ib < nb; ++ib ) {
        batch_atom_index_.push_back(ia);
        batch_local_index_.push_back(ib);
        batch_point_offset_.push_back(running_pt_offset);
        running_pt_offset += batch_npts_cheap_(ia, ib);
      }
    }
    batch_point_offset_.push_back(running_pt_offset);
  }

  size_t natoms()    const noexcept { return atoms_.size(); }
  size_t npts()      const noexcept { return atom_point_offset_.empty() ? 0 : atom_point_offset_.back(); }
  size_t nbatches()  const noexcept { return batch_atom_index_.size(); }

  /// @throws std::out_of_range if `iatom >= natoms()`.
  const AtomInstance<T>& atom( size_t iatom ) const {
    if( iatom >= natoms() ) throw std::out_of_range("MolecularGrid::atom: index out of range");
    return atoms_[iatom];
  }

  size_t atom_npts( size_t iatom ) const {
    if( iatom >= natoms() ) throw std::out_of_range("MolecularGrid::atom_npts: index out of range");
    return atom_batchers_[iatom].npts();
  }
  size_t atom_nbatches( size_t iatom ) const {
    if( iatom >= natoms() ) throw std::out_of_range("MolecularGrid::atom_nbatches: index out of range");
    return atom_batchers_[iatom].nbatches();
  }
  size_t atom_point_begin( size_t iatom ) const {
    if( iatom >= natoms() ) throw std::out_of_range("MolecularGrid::atom_point_begin: index out of range");
    return atom_point_offset_[iatom];
  }
  size_t atom_point_end( size_t iatom ) const {
    if( iatom >= natoms() ) throw std::out_of_range("MolecularGrid::atom_point_end: index out of range");
    return atom_point_offset_[iatom + 1];
  }

  /// Cheap metadata only -- never copies a point/weight vector.
  /// @throws std::out_of_range if `ibatch >= nbatches()`.
  BatchInfo<T> batch_info( size_t ibatch ) const {
    if( ibatch >= nbatches() ) throw std::out_of_range("MolecularGrid::batch_info: index out of range");

    const size_t ia = batch_atom_index_[ibatch];
    const size_t ib = batch_local_index_[ibatch];

    auto it = atom_batchers_[ia].cbegin() + static_cast<int>(ib);
    auto [npts, pb, pe, wb, we] = it.range();
    (void)wb; (void)we;
    auto [box_lo, box_up] = detail::get_box_bounds_points(pb, pe);

    BatchInfo<T> info;
    info.atom_index        = ia;
    info.local_batch_index = ib;
    info.point_begin       = batch_point_offset_[ibatch];
    info.point_end         = batch_point_offset_[ibatch] + npts;
    info.box_lo             = box_lo;
    info.box_up             = box_up;
    return info;
  }

  /// Materializes exactly the requested batch's points/weights.
  /// @throws std::out_of_range if `ibatch >= nbatches()`.
  MolecularGridBatch<T> batch( size_t ibatch ) const {
    if( ibatch >= nbatches() ) throw std::out_of_range("MolecularGrid::batch: index out of range");

    const size_t ia = batch_atom_index_[ibatch];
    const size_t ib = batch_local_index_[ibatch];

    auto [box_lo, box_up, points, weights] = atom_batchers_[ia].at(ib);

    MolecularGridBatch<T> mb;
    mb.atom_index        = ia;
    mb.local_batch_index = ib;
    mb.point_begin       = batch_point_offset_[ibatch];
    mb.point_end         = batch_point_offset_[ibatch + 1];
    mb.box_lo             = box_lo;
    mb.box_up             = box_up;
    mb.points             = std::move(points);
    mb.weights            = std::move(weights);
    return mb;
  }

  /// Materializes exactly the batches belonging to the requested atoms --
  /// nothing else is touched or copied.
  /// @param atom_indices Atom indices whose batches to materialize; may
  ///        contain duplicates or be given in any order.
  /// @throws std::out_of_range if any index is `>= natoms()`.
  std::vector<MolecularGridBatch<T>> batches_for_atoms( const std::vector<size_t>& atom_indices ) const {
    std::vector<MolecularGridBatch<T>> out;
    for( auto ia : atom_indices ) {
      if( ia >= natoms() )
        throw std::out_of_range("MolecularGrid::batches_for_atoms: atom index out of range");
      const size_t global_begin = atom_batch_offset_[ia];
      const size_t global_end   = atom_batch_offset_[ia + 1];
      for( size_t ib = global_begin; ib < global_end; ++ib )
        out.push_back( batch(ib) );
    }
    return out;
  }

  /// Materializes the whole batches that overlap [pt_begin,pt_end) -- a
  /// batch is never split, so a caller wanting an exact array slice should
  /// index points()/weights() directly instead.
  /// @throws std::out_of_range if the range is empty or `pt_end > npts()`.
  std::vector<MolecularGridBatch<T>> batches_for_point_range( size_t pt_begin, size_t pt_end ) const {
    if( pt_begin >= pt_end || pt_end > npts() )
      throw std::out_of_range("MolecularGrid::batches_for_point_range: invalid point range");
    if( nbatches() == 0 ) return {};

    auto first_it = std::upper_bound(
      batch_point_offset_.begin(), batch_point_offset_.end() - 1, pt_begin );
    const size_t ib_first = static_cast<size_t>(
      std::distance(batch_point_offset_.begin(), first_it) - 1 );

    auto last_it = std::upper_bound(
      batch_point_offset_.begin(), batch_point_offset_.end() - 1, pt_end - 1 );
    const size_t ib_last = static_cast<size_t>(
      std::distance(batch_point_offset_.begin(), last_it) - 1 );

    std::vector<MolecularGridBatch<T>> out;
    out.reserve(ib_last - ib_first + 1);
    for( size_t ib = ib_first; ib <= ib_last; ++ib )
      out.push_back( batch(ib) );
    return out;
  }

  /// Lazily materializes (and caches) the full molecular point array on
  /// first call -- not done at construction.
  const point_container& points() const {
    ensure_materialized_();
    return points_cache_;
  }
  /// Lazily materializes (and caches) the full molecular weight array on
  /// first call -- not done at construction.
  const weight_container& weights() const {
    ensure_materialized_();
    return weights_cache_;
  }
  weight_container& weights() {
    ensure_materialized_();
    return weights_cache_;
  }

  /// @brief Apply fuzzy-cell partition weights to the whole molecule, in
  ///        place, forcing full materialization as a precondition.
  /// @throws std::runtime_error if called more than once on the same
  ///         MolecularGrid.
  void apply_partition_weights( PartitionScheme scheme ) {
    if( weights_partitioned_ )
      throw std::runtime_error(
        "MolecularGrid::apply_partition_weights: weights already partitioned");

    ensure_materialized_();
    ensure_geometry_cache_();

    switch(scheme) {
      case PartitionScheme::Becke: reference_becke_partition_weights(*this); break;
      case PartitionScheme::SSF:   reference_ssf_partition_weights(*this);   break;
      case PartitionScheme::LKO:   reference_lko_partition_weights(*this);   break;
    }

    weights_partitioned_ = true;
  }

private:
  std::vector<AtomInstance<T>> atoms_;
  std::vector<batcher_type>    atom_batchers_;

  std::vector<size_t> atom_point_offset_;  // size natoms()+1
  std::vector<size_t> atom_batch_offset_;  // size natoms()+1

  std::vector<size_t> batch_atom_index_;    // size nbatches()
  std::vector<size_t> batch_local_index_;   // size nbatches()
  std::vector<size_t> batch_point_offset_;  // size nbatches()+1

  mutable point_container  points_cache_;
  mutable weight_container weights_cache_;
  mutable bool materialized_ = false;

  std::vector<T> atom_rab_;           // natoms*natoms, row-major
  std::vector<T> atom_dist_nearest_;  // size natoms()
  bool geometry_cache_valid_ = false;
  bool weights_partitioned_  = false;

  size_t batch_npts_cheap_( size_t ia, size_t ib ) const {
    auto it = atom_batchers_[ia].cbegin() + static_cast<int>(ib);
    auto [npts, pb, pe, wb, we] = it.range();
    (void)pb; (void)pe; (void)wb; (void)we;
    return npts;
  }

  void ensure_materialized_() const {
    if( materialized_ ) return;

    points_cache_.clear();
    weights_cache_.clear();
    points_cache_.reserve( npts() );
    weights_cache_.reserve( npts() );

    for( size_t ia = 0; ia < natoms(); ++ia ) {
      const auto& pts = atom_batchers_[ia].points();
      const auto& wts = atom_batchers_[ia].weights();
      points_cache_.insert( points_cache_.end(), pts.begin(), pts.end() );
      weights_cache_.insert( weights_cache_.end(), wts.begin(), wts.end() );
    }

    materialized_ = true;
  }

  void ensure_geometry_cache_() {
    if( geometry_cache_valid_ ) return;

    using traits = fp_traits<T>;

    const size_t na = natoms();
    atom_rab_.assign(na * na, traits::from_integer(0));
    for( size_t i = 0; i < na; ++i )
    for( size_t j = 0; j < i; ++j ) {
      const auto& ci = atoms_[i].center;
      const auto& cj = atoms_[j].center;
      const T dx = ci[0] - cj[0], dy = ci[1] - cj[1], dz = ci[2] - cj[2];
      const T r = traits::sqrt(dx*dx + dy*dy + dz*dz);
      atom_rab_[i + j*na] = r;
      atom_rab_[j + i*na] = r;
    }

    atom_dist_nearest_.assign(na, traits::infinity());
    for( size_t i = 0; i < na; ++i ) {
      T dn = traits::infinity();
      for( size_t j = 0; j < na; ++j )
        if( i != j && atom_rab_[i + j*na] < dn ) dn = atom_rab_[i + j*na];
      atom_dist_nearest_[i] = dn;
    }

    geometry_cache_valid_ = true;
  }

  friend void reference_becke_partition_weights<T>( MolecularGrid<T>& );
  friend void reference_ssf_partition_weights<T>( MolecularGrid<T>& );
  friend void reference_lko_partition_weights<T>( MolecularGrid<T>& );
};

}

// Out-of-line definitions of the reference_*_partition_weights<T> function
// templates declared by partition_weights.hpp (included above), which need
// MolecularGrid<T>'s complete definition (they reach into it via
// friendship) -- included here, after that definition, rather than as a
// trailer on partition_weights.hpp itself, to avoid a circular include
// that would see MolecularGrid<T> only forward-declared. Header-only-safe
// for any T by virtue of being templates.
#include <integratorxx/molecular_grid/impl/partition_weights.hpp>
