#include <integratorxx/molecular_grid/impl/defaults.hpp>

namespace IntegratorXX {

// Explicit instantiation for T = double so the compiled `integratorxx`
// library ships a ready double instantiation for existing consumers.
template struct MolecularGridDefaults<double>;
template std::vector<AtomInstance<double>> make_atom_instances<double>(
  const std::vector<AtomicId>&,
  const std::vector<cartesian_pt_t<double>>&,
  const std::unordered_map<AtomicId, SphericalGridFactory<double>::spherical_grid_ptr>& );

}
