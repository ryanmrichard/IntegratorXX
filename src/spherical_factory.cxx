#include <integratorxx/generators/impl/unpruned_grid.hpp>
#include <integratorxx/generators/impl/pruned_grid.hpp>

namespace IntegratorXX {

// Explicit instantiation for T = double so the compiled `integratorxx`
// library ships a ready double instantiation for existing consumers.
template struct SphericalGridFactory<double>;
template PrunedSphericalGridSpecification<double> robust_psi4_pruning_scheme<double>(
  UnprunedSphericalGridSpecification<double> );
template PrunedSphericalGridSpecification<double> treutler_pruning_scheme<double>(
  UnprunedSphericalGridSpecification<double> );
template PrunedSphericalGridSpecification<double> create_pruned_spec<double>(
  PruningScheme, UnprunedSphericalGridSpecification<double> );

}
