#include <integratorxx/generators/impl/s2_factory.hpp>

namespace IntegratorXX {

// Explicit instantiation for T = double so the compiled `integratorxx`
// library ships a ready double instantiation for existing consumers.
template struct S2Factory<double>;

}
