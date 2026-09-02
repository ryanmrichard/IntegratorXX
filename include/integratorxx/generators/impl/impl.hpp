#include <integratorxx/generators/impl/unpruned_grid.hpp>
#include <integratorxx/generators/impl/pruned_grid.hpp>
#include <integratorxx/generators/impl/radial_factory.hpp>
#include <integratorxx/generators/impl/s2_factory.hpp>

// Runtime pieces of the molecular_grid API (see
// integratorxx/molecular_grid/), aggregated here alongside the rest of the
// non-template runtime generator layer so that a single
// `#include <integratorxx/generators/impl/impl.hpp>` (exactly once per
// project) is enough for a full header-only build.
#include <integratorxx/molecular_grid/impl/defaults.hpp>
#include <integratorxx/molecular_grid/impl/partition_weights.hpp>

