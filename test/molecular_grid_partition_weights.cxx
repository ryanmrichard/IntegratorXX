#include "catch2/catch_all.hpp"

#include <integratorxx/molecular_grid/molecular_grid.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>
#include <integratorxx/generators/spherical_factory.hpp>

#include <cmath>
#include <limits>

using namespace IntegratorXX;

namespace {

/// Comparison tolerance scaled for the type under test: `float`'s ~7
/// significant decimal digits can't meet a tolerance appropriate for
/// `double`.
template <typename T>
constexpr T partition_tol() { return std::is_same_v<T,float> ? T(1e-4) : T(1e-10); }

template <typename T>
using mk_type_t  = MuraKnowles<T,T>;
template <typename T>
using ll_type_t  = LebedevLaikov<T>;
template <typename T>
using sph_type_t = SphericalQuadrature<mk_type_t<T>, ll_type_t<T>>;

template <typename T>
typename SphericalGridFactory<T>::spherical_grid_ptr
  make_test_grid_ptr( size_t nrad = 10, size_t nang = 26 ) {
  mk_type_t<T> rq(nrad, T(2.0));
  ll_type_t<T> aq(nang);
  return std::make_shared<sph_type_t<T>>( sph_type_t<T>(rq, aq) );
}

// Independently reimplemented (not sharing code with impl/partition_weights.hpp)
// Becke/SSF cell functions and per-point fraction evaluators, used as an
// oracle to cross-check the ported implementation against real grid points.

template <typename T>
T oracle_dist( const cartesian_pt_t<T>& a, const cartesian_pt_t<T>& b ) {
  const T dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

template <typename T>
T oracle_hBecke( T x ) { return T(1.5)*x - T(0.5)*x*x*x; }
template <typename T>
T oracle_gBecke( T x ) { return oracle_hBecke(oracle_hBecke(oracle_hBecke(x))); }

template <typename T>
T oracle_becke_fraction(
  const std::vector<cartesian_pt_t<T>>& centers, size_t home, const cartesian_pt_t<T>& pt
) {
  const size_t n = centers.size();
  std::vector<T> atomDist(n);
  for( size_t i = 0; i < n; ++i ) atomDist[i] = oracle_dist<T>(pt, centers[i]);

  std::vector<T> P(n, T(1.0));
  for( size_t i = 0; i < n; ++i )
  for( size_t j = 0; j < i; ++j ) {
    const T Rij = oracle_dist<T>(centers[i], centers[j]);
    const T g   = oracle_gBecke<T>( (atomDist[i]-atomDist[j]) / Rij );
    P[i] *= T(0.5)*(T(1.)-g);
    P[j] *= T(0.5)*(T(1.)+g);
  }
  T sum = T(0.);
  for( auto p : P ) sum += p;
  return P[home] / sum;
}

template <typename T>
T oracle_gFrisch( T x ) {
  const T s  = x / T(0.64);
  const T s2 = s*s, s3 = s*s2, s5 = s3*s2, s7 = s5*s2;
  return (T(35.)*(s - s3) + T(21.)*s5 - T(5.)*s7) / T(16.);
}

template <typename T>
T oracle_ssf_fraction(
  const std::vector<cartesian_pt_t<T>>& centers, size_t home,
  const cartesian_pt_t<T>& pt, T dist_nearest_home
) {
  const size_t n = centers.size();
  const T dist_cutoff = T(0.5) * (T(1.) - T(0.64)) * dist_nearest_home;

  std::vector<T> atomDist(n);
  for( size_t i = 0; i < n; ++i ) atomDist[i] = oracle_dist<T>(pt, centers[i]);
  if( atomDist[home] < dist_cutoff ) return T(1.0);

  std::vector<T> P(n, T(1.0));
  for( size_t i = 0; i < n; ++i )
  for( size_t j = 0; j < i; ++j ) {
    const T Rij = oracle_dist<T>(centers[i], centers[j]);
    const T mu  = (atomDist[i]-atomDist[j]) / Rij;
    if( mu <= T(-0.64) )      P[j] = T(0.);
    else if( mu >= T(0.64) )  P[i] = T(0.);
    else {
      const T g = T(0.5)*(T(1.) - oracle_gFrisch<T>(mu));
      P[i] *= g;
      P[j] *= T(1.) - g;
    }
  }
  T sum = T(0.);
  for( auto p : P ) sum += p;
  return P[home] / sum;
}

}

TEMPLATE_TEST_CASE( "Partition Weights Oracle Cross-Check", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.0), T(0.0), T(0.0)}, make_test_grid_ptr<T>(10, 26) },
    AtomInstance<T>{ {T(2.5), T(0.0), T(0.0)}, make_test_grid_ptr<T>(12, 50) },
    AtomInstance<T>{ {T(0.0), T(3.0), T(0.0)}, make_test_grid_ptr<T>(8,  26) },
  };
  std::vector<cartesian_pt_t<T>> centers = { atoms[0].center, atoms[1].center, atoms[2].center };

  std::vector<T> dist_nearest(3);
  for( size_t i = 0; i < 3; ++i ) {
    T dn = std::numeric_limits<T>::infinity();
    for( size_t j = 0; j < 3; ++j ) if( i != j ) dn = std::min(dn, oracle_dist<T>(centers[i], centers[j]));
    dist_nearest[i] = dn;
  }

  SECTION("Becke") {
    MolecularGrid<T> mg( atoms, 30 );
    const auto before        = mg.weights();
    const auto before_points = mg.points();
    mg.apply_partition_weights( PartitionScheme::Becke );
    const auto& after = mg.weights();

    for( size_t ia = 0; ia < mg.natoms(); ++ia )
    for( size_t ip = mg.atom_point_begin(ia); ip < mg.atom_point_end(ia); ++ip ) {
      const T expected = oracle_becke_fraction<T>(centers, ia, before_points[ip]);
      const T actual   = after[ip] / before[ip];
      REQUIRE_THAT( actual, Catch::Matchers::WithinAbs(expected, partition_tol<T>()) );
    }
  }

  SECTION("SSF") {
    MolecularGrid<T> mg( atoms, 30 );
    const auto before        = mg.weights();
    const auto before_points = mg.points();
    mg.apply_partition_weights( PartitionScheme::SSF );
    const auto& after = mg.weights();

    for( size_t ia = 0; ia < mg.natoms(); ++ia )
    for( size_t ip = mg.atom_point_begin(ia); ip < mg.atom_point_end(ia); ++ip ) {
      const T expected = oracle_ssf_fraction<T>(centers, ia, before_points[ip], dist_nearest[ia]);
      const T actual   = after[ip] / before[ip];
      REQUIRE_THAT( actual, Catch::Matchers::WithinAbs(expected, partition_tol<T>()) );
    }
  }
}

TEMPLATE_TEST_CASE( "Becke/LKO partition fractions stay within [0,1] for axis-aligned points", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  // A grid point exactly on the axis joining two atoms gives the Becke
  // switching coordinate mu = +-1 exactly, which is the boundary case naive
  // interval arithmetic's dependency problem can round outside of if
  // hBecke/gBecke/gFrisch aren't re-clamped into their provably-exact
  // [-1,1] range after each (self-correlated) evaluation -- see
  // fp_traits<T>::clamp and its callers in impl/partition_weights.hpp. A
  // degree-6 Lebedev-Laikov angular grid places points exactly at
  // (0,0,+-1), so aligning both atoms on the z-axis reproduces that
  // boundary deterministically. This is a double/float-only regression
  // (partitionScratch/mu are plain scalars here, not an enclosure), but it
  // still exercises the new clamp call on every hBecke/gFrisch evaluation.
  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.0), T(0.0), T(0.0)}, make_test_grid_ptr<T>(5, 6) },
    AtomInstance<T>{ {T(0.0), T(0.0), T(1.3984)}, make_test_grid_ptr<T>(5, 6) },
  };

  auto check = [&]( PartitionScheme scheme ) {
    auto atoms_copy = atoms;
    MolecularGrid<T> mg( atoms_copy, 512 );
    const auto before = mg.weights();
    mg.apply_partition_weights( scheme );
    const auto& after = mg.weights();
    for( size_t i = 0; i < after.size(); ++i ) {
      const T fraction = after[i] / before[i];
      REQUIRE( fraction >= T(0) );
      REQUIRE( fraction <= T(1) );
    }
  };

  SECTION("Becke") { check(PartitionScheme::Becke); }
  SECTION("LKO")   { check(PartitionScheme::LKO); }
}

TEMPLATE_TEST_CASE( "LKO equals Becke for a compact molecule", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  // All pairwise distances well under the 5.0-bohr LKO cutoff, so pruning
  // is a no-op and LKO should reduce exactly to Becke.
  std::vector<AtomInstance<T>> atoms_becke = {
    AtomInstance<T>{ {T(0.0), T(0.0), T(0.0)}, make_test_grid_ptr<T>(10, 26) },
    AtomInstance<T>{ {T(2.0), T(0.0), T(0.0)}, make_test_grid_ptr<T>(10, 26) },
    AtomInstance<T>{ {T(0.0), T(2.0), T(0.0)}, make_test_grid_ptr<T>(10, 26) },
  };
  auto atoms_lko = atoms_becke;

  MolecularGrid<T> mg_becke( atoms_becke, 30 );
  mg_becke.apply_partition_weights( PartitionScheme::Becke );

  MolecularGrid<T> mg_lko( atoms_lko, 30 );
  mg_lko.apply_partition_weights( PartitionScheme::LKO );

  REQUIRE( mg_becke.npts() == mg_lko.npts() );
  const auto& wb = mg_becke.weights();
  const auto& wl = mg_lko.weights();
  for( size_t i = 0; i < wb.size(); ++i )
    REQUIRE_THAT( wl[i], Catch::Matchers::WithinAbs(wb[i], partition_tol<T>()) );
}

TEMPLATE_TEST_CASE( "LKO near-field pruning ignores a far-away atom", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  auto g0 = make_test_grid_ptr<T>(10, 26);
  auto g1 = make_test_grid_ptr<T>(10, 26);

  std::vector<AtomInstance<T>> pair_atoms = {
    AtomInstance<T>{ {T(0.0), T(0.0), T(0.0)}, g0 },
    AtomInstance<T>{ {T(2.0), T(0.0), T(0.0)}, g1 },
  };
  MolecularGrid<T> mg_pair( pair_atoms, 30 );
  mg_pair.apply_partition_weights( PartitionScheme::LKO );

  std::vector<AtomInstance<T>> triple_atoms = {
    AtomInstance<T>{ {T(0.0),   T(0.0), T(0.0)}, g0 },
    AtomInstance<T>{ {T(2.0),   T(0.0), T(0.0)}, g1 },
    AtomInstance<T>{ {T(100.0), T(0.0), T(0.0)}, make_test_grid_ptr<T>(10, 26) }, // far beyond r_nearest + R_cutoff
  };
  MolecularGrid<T> mg_triple( triple_atoms, 30 );
  mg_triple.apply_partition_weights( PartitionScheme::LKO );

  const size_t pair_npts = mg_pair.npts();
  REQUIRE( mg_triple.atom_point_end(1) == pair_npts );

  const auto& w_pair   = mg_pair.weights();
  const auto& w_triple = mg_triple.weights();
  for( size_t i = 0; i < pair_npts; ++i )
    REQUIRE_THAT( w_triple[i], Catch::Matchers::WithinAbs(w_pair[i], partition_tol<T>()) );
}

TEMPLATE_TEST_CASE( "apply_partition_weights throws on double-apply", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  auto g = make_test_grid_ptr<T>();
  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.), T(0.), T(0.)}, g },
    AtomInstance<T>{ {T(3.), T(0.), T(0.)}, g },
  };
  MolecularGrid<T> mg( atoms, 30 );
  mg.apply_partition_weights( PartitionScheme::Becke );
  REQUIRE_THROWS_AS( mg.apply_partition_weights( PartitionScheme::Becke ), std::runtime_error );
}

TEMPLATE_TEST_CASE( "SSF dist_cutoff leaves near-nucleus weights exactly unchanged", "[molecular-grid][weights]", double, float ) {
  using T = TestType;

  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.), T(0.), T(0.)}, make_test_grid_ptr<T>(20, 26) },
    AtomInstance<T>{ {T(6.), T(0.), T(0.)}, make_test_grid_ptr<T>(20, 26) },
  };

  MolecularGrid<T> mg( atoms, 30 );
  const auto before        = mg.weights();
  const auto before_points = mg.points();
  mg.apply_partition_weights( PartitionScheme::SSF );
  const auto& after = mg.weights();

  const T dist_nearest0 = T(6.0);
  const T dist_cutoff   = T(0.5) * (T(1.) - T(0.64)) * dist_nearest0;

  bool found_a_cutoff_point = false;
  for( size_t ip = mg.atom_point_begin(0); ip < mg.atom_point_end(0); ++ip ) {
    const auto& p = before_points[ip];
    const T d0 = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]); // atom 0 is at the origin
    if( d0 < dist_cutoff ) {
      found_a_cutoff_point = true;
      REQUIRE( after[ip] == before[ip] ); // untouched: exact equality, not approximate
    }
  }
  REQUIRE( found_a_cutoff_point ); // sanity: the test actually exercised the cutoff branch
}
