#include <integratorxx/generators/impl/radial_factory.hpp>

namespace IntegratorXX {

// Explicit instantiation for T = double so the compiled `integratorxx`
// library ships a ready double instantiation for existing consumers.
template struct RadialFactory<double>;

}
