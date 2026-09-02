#include "catch2/catch_all.hpp"

#include <integratorxx/molecular_grid/molecular_grid.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>
#include <integratorxx/generators/spherical_factory.hpp>

#include <cmath>
#include <limits>

using namespace IntegratorXX;

using mk_type  = MuraKnowles<double,double>;
using ll_type  = LebedevLaikov<double>;
using sph_type = SphericalQuadrature<mk_type, ll_type>;

namespace {

SphericalGridFactory::spherical_grid_ptr make_test_grid_ptr( size_t nrad = 10, size_t nang = 26 ) {
  mk_type rq(nrad, 2.0);
  ll_type aq(nang);
  return std::make_shared<sph_type>( sph_type(rq, aq) );
}

// Independently reimplemented (not sharing code with impl/partition_weights.hpp)
// Becke/SSF cell functions and per-point fraction evaluators, used as an
// oracle to cross-check the ported implementation against real grid points.

double oracle_dist( const cartesian_pt_t<double>& a, const cartesian_pt_t<double>& b ) {
  const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double oracle_hBecke( double x ) { return 1.5*x - 0.5*x*x*x; }
double oracle_gBecke( double x ) { return oracle_hBecke(oracle_hBecke(oracle_hBecke(x))); }

double oracle_becke_fraction(
  const std::vector<cartesian_pt_t<double>>& centers, size_t home, const cartesian_pt_t<double>& pt
) {
  const size_t n = centers.size();
  std::vector<double> atomDist(n);
  for( size_t i = 0; i < n; ++i ) atomDist[i] = oracle_dist(pt, centers[i]);

  std::vector<double> P(n, 1.0);
  for( size_t i = 0; i < n; ++i )
  for( size_t j = 0; j < i; ++j ) {
    const double Rij = oracle_dist(centers[i], centers[j]);
    const double g   = oracle_gBecke( (atomDist[i]-atomDist[j]) / Rij );
    P[i] *= 0.5*(1.-g);
    P[j] *= 0.5*(1.+g);
  }
  double sum = 0.;
  for( auto p : P ) sum += p;
  return P[home] / sum;
}

double oracle_gFrisch( double x ) {
  const double s  = x / 0.64;
  const double s2 = s*s, s3 = s*s2, s5 = s3*s2, s7 = s5*s2;
  return (35.*(s - s3) + 21.*s5 - 5.*s7) / 16.;
}

double oracle_ssf_fraction(
  const std::vector<cartesian_pt_t<double>>& centers, size_t home,
  const cartesian_pt_t<double>& pt, double dist_nearest_home
) {
  const size_t n = centers.size();
  const double dist_cutoff = 0.5 * (1. - 0.64) * dist_nearest_home;

  std::vector<double> atomDist(n);
  for( size_t i = 0; i < n; ++i ) atomDist[i] = oracle_dist(pt, centers[i]);
  if( atomDist[home] < dist_cutoff ) return 1.0;

  std::vector<double> P(n, 1.0);
  for( size_t i = 0; i < n; ++i )
  for( size_t j = 0; j < i; ++j ) {
    const double Rij = oracle_dist(centers[i], centers[j]);
    const double mu  = (atomDist[i]-atomDist[j]) / Rij;
    if( mu <= -0.64 )      P[j] = 0.;
    else if( mu >= 0.64 )  P[i] = 0.;
    else {
      const double g = 0.5*(1. - oracle_gFrisch(mu));
      P[i] *= g;
      P[j] *= 1. - g;
    }
  }
  double sum = 0.;
  for( auto p : P ) sum += p;
  return P[home] / sum;
}

}

TEST_CASE( "Partition Weights Oracle Cross-Check", "[molecular-grid][weights]" ) {

  std::vector<AtomInstance> atoms = {
    AtomInstance{ {0.0, 0.0, 0.0}, make_test_grid_ptr(10, 26) },
    AtomInstance{ {2.5, 0.0, 0.0}, make_test_grid_ptr(12, 50) },
    AtomInstance{ {0.0, 3.0, 0.0}, make_test_grid_ptr(8,  26) },
  };
  std::vector<cartesian_pt_t<double>> centers = { atoms[0].center, atoms[1].center, atoms[2].center };

  std::vector<double> dist_nearest(3);
  for( size_t i = 0; i < 3; ++i ) {
    double dn = std::numeric_limits<double>::infinity();
    for( size_t j = 0; j < 3; ++j ) if( i != j ) dn = std::min(dn, oracle_dist(centers[i], centers[j]));
    dist_nearest[i] = dn;
  }

  SECTION("Becke") {
    MolecularGrid mg( atoms, 30 );
    const auto before        = mg.weights();
    const auto before_points = mg.points();
    mg.apply_partition_weights( PartitionScheme::Becke );
    const auto& after = mg.weights();

    for( size_t ia = 0; ia < mg.natoms(); ++ia )
    for( size_t ip = mg.atom_point_begin(ia); ip < mg.atom_point_end(ia); ++ip ) {
      const double expected = oracle_becke_fraction(centers, ia, before_points[ip]);
      const double actual   = after[ip] / before[ip];
      REQUIRE_THAT( actual, Catch::Matchers::WithinAbs(expected, 1e-10) );
    }
  }

  SECTION("SSF") {
    MolecularGrid mg( atoms, 30 );
    const auto before        = mg.weights();
    const auto before_points = mg.points();
    mg.apply_partition_weights( PartitionScheme::SSF );
    const auto& after = mg.weights();

    for( size_t ia = 0; ia < mg.natoms(); ++ia )
    for( size_t ip = mg.atom_point_begin(ia); ip < mg.atom_point_end(ia); ++ip ) {
      const double expected = oracle_ssf_fraction(centers, ia, before_points[ip], dist_nearest[ia]);
      const double actual   = after[ip] / before[ip];
      REQUIRE_THAT( actual, Catch::Matchers::WithinAbs(expected, 1e-10) );
    }
  }
}

TEST_CASE( "LKO equals Becke for a compact molecule", "[molecular-grid][weights]" ) {

  // All pairwise distances well under the 5.0-bohr LKO cutoff, so pruning
  // is a no-op and LKO should reduce exactly to Becke.
  std::vector<AtomInstance> atoms_becke = {
    AtomInstance{ {0.0, 0.0, 0.0}, make_test_grid_ptr(10, 26) },
    AtomInstance{ {2.0, 0.0, 0.0}, make_test_grid_ptr(10, 26) },
    AtomInstance{ {0.0, 2.0, 0.0}, make_test_grid_ptr(10, 26) },
  };
  auto atoms_lko = atoms_becke;

  MolecularGrid mg_becke( atoms_becke, 30 );
  mg_becke.apply_partition_weights( PartitionScheme::Becke );

  MolecularGrid mg_lko( atoms_lko, 30 );
  mg_lko.apply_partition_weights( PartitionScheme::LKO );

  REQUIRE( mg_becke.npts() == mg_lko.npts() );
  const auto& wb = mg_becke.weights();
  const auto& wl = mg_lko.weights();
  for( size_t i = 0; i < wb.size(); ++i )
    REQUIRE_THAT( wl[i], Catch::Matchers::WithinAbs(wb[i], 1e-10) );
}

TEST_CASE( "LKO near-field pruning ignores a far-away atom", "[molecular-grid][weights]" ) {

  auto g0 = make_test_grid_ptr(10, 26);
  auto g1 = make_test_grid_ptr(10, 26);

  std::vector<AtomInstance> pair_atoms = {
    AtomInstance{ {0.0, 0.0, 0.0}, g0 },
    AtomInstance{ {2.0, 0.0, 0.0}, g1 },
  };
  MolecularGrid mg_pair( pair_atoms, 30 );
  mg_pair.apply_partition_weights( PartitionScheme::LKO );

  std::vector<AtomInstance> triple_atoms = {
    AtomInstance{ {0.0,   0.0, 0.0}, g0 },
    AtomInstance{ {2.0,   0.0, 0.0}, g1 },
    AtomInstance{ {100.0, 0.0, 0.0}, make_test_grid_ptr(10, 26) }, // far beyond r_nearest + R_cutoff
  };
  MolecularGrid mg_triple( triple_atoms, 30 );
  mg_triple.apply_partition_weights( PartitionScheme::LKO );

  const size_t pair_npts = mg_pair.npts();
  REQUIRE( mg_triple.atom_point_end(1) == pair_npts );

  const auto& w_pair   = mg_pair.weights();
  const auto& w_triple = mg_triple.weights();
  for( size_t i = 0; i < pair_npts; ++i )
    REQUIRE_THAT( w_triple[i], Catch::Matchers::WithinAbs(w_pair[i], 1e-10) );
}

TEST_CASE( "apply_partition_weights throws on double-apply", "[molecular-grid][weights]" ) {

  auto g = make_test_grid_ptr();
  std::vector<AtomInstance> atoms = {
    AtomInstance{ {0., 0., 0.}, g },
    AtomInstance{ {3., 0., 0.}, g },
  };
  MolecularGrid mg( atoms, 30 );
  mg.apply_partition_weights( PartitionScheme::Becke );
  REQUIRE_THROWS_AS( mg.apply_partition_weights( PartitionScheme::Becke ), std::runtime_error );
}

TEST_CASE( "SSF dist_cutoff leaves near-nucleus weights exactly unchanged", "[molecular-grid][weights]" ) {

  std::vector<AtomInstance> atoms = {
    AtomInstance{ {0., 0., 0.}, make_test_grid_ptr(20, 26) },
    AtomInstance{ {6., 0., 0.}, make_test_grid_ptr(20, 26) },
  };

  MolecularGrid mg( atoms, 30 );
  const auto before        = mg.weights();
  const auto before_points = mg.points();
  mg.apply_partition_weights( PartitionScheme::SSF );
  const auto& after = mg.weights();

  const double dist_nearest0 = 6.0;
  const double dist_cutoff   = 0.5 * (1. - 0.64) * dist_nearest0;

  bool found_a_cutoff_point = false;
  for( size_t ip = mg.atom_point_begin(0); ip < mg.atom_point_end(0); ++ip ) {
    const auto& p = before_points[ip];
    const double d0 = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]); // atom 0 is at the origin
    if( d0 < dist_cutoff ) {
      found_a_cutoff_point = true;
      REQUIRE( after[ip] == before[ip] ); // untouched: exact equality, not approximate
    }
  }
  REQUIRE( found_a_cutoff_point ); // sanity: the test actually exercised the cutoff branch
}
