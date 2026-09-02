#include <integratorxx/molecular_grid/impl/partition_weights.hpp>

namespace IntegratorXX {

// Explicit instantiation for T = double so the compiled `integratorxx`
// library ships a ready double instantiation for existing consumers.
template void reference_becke_partition_weights<double>( MolecularGrid<double>& );
template void reference_ssf_partition_weights<double>( MolecularGrid<double>& );
template void reference_lko_partition_weights<double>( MolecularGrid<double>& );

}
