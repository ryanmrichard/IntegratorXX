#include "catch2/catch_all.hpp"

#include <integratorxx/molecular_grid/molecular_grid.hpp>
#include <integratorxx/molecular_grid/defaults.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>
#include <integratorxx/generators/spherical_factory.hpp>

using namespace IntegratorXX;

namespace {

/// Comparison tolerances scaled for the type under test: `float`'s ~7
/// significant decimal digits can't meet the tolerances that are
/// appropriate for `double`, so both a "tight" (per-point/per-weight) and
/// "loose" (accumulated-sum) tolerance are widened for `float`.
template <typename T>
constexpr T tight_tol() { return std::is_same_v<T,float> ? T(1e-4) : T(1e-13); }
template <typename T>
constexpr T loose_tol() { return std::is_same_v<T,float> ? T(1e-4) : T(1e-10); }

template <typename T>
using mk_type_t  = MuraKnowles<T,T>;
template <typename T>
using ll_type_t  = LebedevLaikov<T>;
template <typename T>
using sph_type_t = SphericalQuadrature<mk_type_t<T>, ll_type_t<T>>;

template <typename T>
sph_type_t<T> make_test_atomic_grid( size_t nrad = 10, size_t nang = 26 ) {
  mk_type_t<T> rq(nrad, T(2.0));
  ll_type_t<T> aq(nang);
  return sph_type_t<T>(rq, aq);
}

template <typename T>
typename SphericalGridFactory<T>::spherical_grid_ptr
  make_test_grid_ptr( size_t nrad = 10, size_t nang = 26 ) {
  return std::make_shared<sph_type_t<T>>( make_test_atomic_grid<T>(nrad, nang) );
}

template <typename T>
void require_points_equal( const cartesian_pt_t<T>& a, const cartesian_pt_t<T>& b,
  T tol = tight_tol<T>() ) {
  REQUIRE_THAT( a[0], Catch::Matchers::WithinAbs(b[0], tol) );
  REQUIRE_THAT( a[1], Catch::Matchers::WithinAbs(b[1], tol) );
  REQUIRE_THAT( a[2], Catch::Matchers::WithinAbs(b[2], tol) );
}

/// Test-only decorator: counts clone() calls so construction's "clone each
/// atom's template exactly once" contract is directly verifiable.
template <typename T>
struct CountingSphericalQuadrature : public sph_type_t<T> {
  using base_type = sph_type_t<T>;
  int* counter = nullptr;

  explicit CountingSphericalQuadrature( const sph_type_t<T>& q ) : base_type(q) {}

  std::shared_ptr<SphericalQuadratureBase<typename base_type::point_container,typename base_type::weight_container>>
    clone() const override {
    if( counter ) ++(*counter);
    return std::make_shared<CountingSphericalQuadrature>(*this);
  }
};

}

TEMPLATE_TEST_CASE( "MolecularGrid Construction", "[molecular-grid]", double, float ) {
  using T = TestType;

  auto grid_ptr = make_test_grid_ptr<T>();
  const size_t atomic_npts = grid_ptr->npts();

  SECTION("Two atoms, shared template") {
    std::vector<AtomInstance<T>> atoms = {
      AtomInstance<T>{ {T(0.), T(0.), T(0.)}, grid_ptr },
      AtomInstance<T>{ {T(5.), T(0.), T(0.)}, grid_ptr }
    };

    const auto orig_points = grid_ptr->points();
    const auto orig_center = grid_ptr->center();

    MolecularGrid<T> mg( atoms, 50 );

    REQUIRE( mg.natoms() == 2 );
    REQUIRE( mg.npts() == 2*atomic_npts );
    REQUIRE( mg.atom_npts(0) == atomic_npts );
    REQUIRE( mg.atom_npts(1) == atomic_npts );
    REQUIRE( mg.atom_point_begin(0) == 0 );
    REQUIRE( mg.atom_point_end(0)   == atomic_npts );
    REQUIRE( mg.atom_point_begin(1) == atomic_npts );
    REQUIRE( mg.atom_point_end(1)   == 2*atomic_npts );

    // The shared template must be unmutated by construction.
    require_points_equal<T>( grid_ptr->center(), orig_center );
    REQUIRE( grid_ptr->points().size() == orig_points.size() );
    for( size_t i = 0; i < orig_points.size(); ++i )
      require_points_equal<T>( grid_ptr->points()[i], orig_points[i] );
  }

  SECTION("Clone economy: exactly one clone() per atom, only at construction") {
    int clone_count = 0;
    CountingSphericalQuadrature<T> template_quad( make_test_atomic_grid<T>() );
    template_quad.counter = &clone_count;
    auto template_ptr = std::make_shared<CountingSphericalQuadrature<T>>(template_quad);

    std::vector<AtomInstance<T>> atoms = {
      AtomInstance<T>{ {T(0.), T(0.), T(0.)}, template_ptr },
      AtomInstance<T>{ {T(3.), T(0.), T(0.)}, template_ptr },
      AtomInstance<T>{ {T(6.), T(0.), T(0.)}, template_ptr },
    };

    MolecularGrid<T> mg( atoms, 50 );
    REQUIRE( clone_count == 3 );

    // Cheap accessors must not trigger additional clones.
    (void)mg.natoms(); (void)mg.npts(); (void)mg.nbatches();
    for( size_t ib = 0; ib < mg.nbatches(); ++ib ) (void)mg.batch_info(ib);
    REQUIRE( clone_count == 3 );

    // Nor must materializing accessors (they read out of the clones already
    // made at construction; they don't clone again).
    (void)mg.batches_for_atoms({0});
    (void)mg.points();
    REQUIRE( clone_count == 3 );
  }
}

TEMPLATE_TEST_CASE( "MolecularGrid Batch Retrieval Consistency", "[molecular-grid]", double, float ) {
  using T = TestType;

  auto grid_ptr = make_test_grid_ptr<T>(12, 26);

  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.), T(0.),  T(0.)}, grid_ptr },
    AtomInstance<T>{ {T(4.), T(0.),  T(0.)}, grid_ptr },
    AtomInstance<T>{ {T(0.), T(4.),  T(0.)}, grid_ptr },
  };

  MolecularGrid<T> mg( atoms, 20 );
  REQUIRE( mg.natoms() == 3 );
  REQUIRE( mg.nbatches() > 0 );

  SECTION("batches_for_atoms matches an independent clone+recenter+batch reference") {
    const size_t ia = 1;

    auto ref_clone = atoms[ia].grid->clone();
    ref_clone->recenter( atoms[ia].center );
    SphericalMicroBatcher<std::vector<cartesian_pt_t<T>>, std::vector<T>>
      ref_batcher( 20, ref_clone );

    auto batches = mg.batches_for_atoms({ia});
    REQUIRE( batches.size() == ref_batcher.nbatches() );

    for( size_t ib = 0; ib < batches.size(); ++ib ) {
      auto [box_lo, box_up, ref_pts, ref_wts] = ref_batcher.at(ib);
      REQUIRE( batches[ib].points.size() == ref_pts.size() );
      for( size_t ip = 0; ip < ref_pts.size(); ++ip ) {
        require_points_equal<T>( batches[ib].points[ip], ref_pts[ip] );
        REQUIRE_THAT( batches[ib].weights[ip], Catch::Matchers::WithinAbs(ref_wts[ip], tight_tol<T>()) );
      }
    }
  }

  SECTION("Flat list filtered by atom boundary equals batches_for_atoms concatenated") {
    for( size_t ia = 0; ia < mg.natoms(); ++ia ) {
      auto batches = mg.batches_for_atoms({ia});

      std::vector<cartesian_pt_t<T>> concat_pts;
      std::vector<T> concat_wts;
      for( auto& b : batches ) {
        concat_pts.insert(concat_pts.end(), b.points.begin(), b.points.end());
        concat_wts.insert(concat_wts.end(), b.weights.begin(), b.weights.end());
      }

      const auto pt_begin = mg.atom_point_begin(ia);
      const auto pt_end   = mg.atom_point_end(ia);
      REQUIRE( concat_pts.size() == (pt_end - pt_begin) );

      const auto& all_pts = mg.points();
      const auto& all_wts = mg.weights();
      for( size_t i = 0; i < concat_pts.size(); ++i ) {
        require_points_equal<T>( concat_pts[i], all_pts[pt_begin + i] );
        REQUIRE_THAT( concat_wts[i], Catch::Matchers::WithinAbs(all_wts[pt_begin + i], tight_tol<T>()) );
      }
    }
  }

  SECTION("batches_for_point_range covers the requested range and matches the flat list") {
    const auto& all_pts = mg.points();
    const auto& all_wts = mg.weights();

    std::vector<std::pair<size_t,size_t>> windows = {
      {0, mg.npts()},
      {0, mg.atom_point_end(0)},                       // exactly the first atom
      {mg.atom_point_begin(1) + 1, mg.atom_point_end(1) - 1}, // straddles internal batch boundaries
      {mg.atom_point_end(0) - 1, mg.atom_point_begin(1) + 1}, // straddles an atom boundary
    };

    for( auto [pb, pe] : windows ) {
      auto batches = mg.batches_for_point_range(pb, pe);
      REQUIRE( !batches.empty() );

      size_t cover_begin = batches.front().point_begin;
      size_t cover_end   = batches.back().point_end;
      REQUIRE( cover_begin <= pb );
      REQUIRE( cover_end   >= pe );

      for( auto& b : batches ) {
        for( size_t i = 0; i < b.points.size(); ++i ) {
          const size_t gi = b.point_begin + i;
          require_points_equal<T>( b.points[i], all_pts[gi] );
          REQUIRE_THAT( b.weights[i], Catch::Matchers::WithinAbs(all_wts[gi], tight_tol<T>()) );
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE( "MolecularGrid Hetero Round Trip", "[molecular-grid]", double, float ) {
  using T = TestType;

  // A toy water-like molecule: two atoms share one "H" template, one atom
  // uses a distinct "O" template.
  auto h_grid = make_test_grid_ptr<T>(8, 26);
  auto o_grid = make_test_grid_ptr<T>(14, 50);

  std::vector<AtomInstance<T>> atoms = {
    AtomInstance<T>{ {T(0.0),  T(0.0), T(0.0)}, o_grid },
    AtomInstance<T>{ {T(1.5),  T(1.0), T(0.0)}, h_grid },
    AtomInstance<T>{ {T(-1.5), T(1.0), T(0.0)}, h_grid },
  };

  MolecularGrid<T> mg( atoms, 30 );

  REQUIRE( mg.natoms() == 3 );
  REQUIRE( mg.npts() == o_grid->npts() + 2*h_grid->npts() );

  // Recentering shifts points, never weights -- total weight is invariant.
  T ref_weight_sum = T(0.);
  for( auto w : o_grid->weights() ) ref_weight_sum += w;
  for( auto w : h_grid->weights() ) ref_weight_sum += T(2.) * w;

  T weight_sum = T(0.);
  for( auto w : mg.weights() ) weight_sum += w;
  // A relative tolerance here (rather than the absolute tight_tol/loose_tol
  // used elsewhere) since this sum's magnitude (hundreds) swamps an
  // absolute per-point tolerance.
  REQUIRE_THAT( weight_sum, Catch::Matchers::WithinRel(ref_weight_sum, loose_tol<T>()) );

  // Full-molecule atom subset reproduces the flat list exactly.
  auto all_batches = mg.batches_for_atoms({0, 1, 2});
  std::vector<cartesian_pt_t<T>> concat_pts;
  std::vector<T> concat_wts;
  for( auto& b : all_batches ) {
    concat_pts.insert(concat_pts.end(), b.points.begin(), b.points.end());
    concat_wts.insert(concat_wts.end(), b.weights.begin(), b.weights.end());
  }
  REQUIRE( concat_pts.size() == mg.npts() );
  const auto& flat_pts = mg.points();
  const auto& flat_wts = mg.weights();
  for( size_t i = 0; i < concat_pts.size(); ++i ) {
    require_points_equal<T>( concat_pts[i], flat_pts[i] );
    REQUIRE_THAT( concat_wts[i], Catch::Matchers::WithinAbs(flat_wts[i], tight_tol<T>()) );
  }
}

TEMPLATE_TEST_CASE( "MolecularGridDefaults Sanity", "[molecular-grid]", double, float ) {
  using T = TestType;

  const std::vector<RadialQuad> radial_quads = {
    RadialQuad::MuraKnowles, RadialQuad::MurrayHandyLaming, RadialQuad::TreutlerAhlrichs
  };
  const std::vector<AtomicGridSizeDefault> presets = {
    AtomicGridSizeDefault::GM3, AtomicGridSizeDefault::GM5,
    AtomicGridSizeDefault::FineGrid, AtomicGridSizeDefault::UltraFineGrid,
    AtomicGridSizeDefault::SuperFineGrid
  };

  for( AtomicId Z = 1; Z <= 118; ++Z ) {
    REQUIRE( default_atomic_radius(Z) > 0. );

    for( auto preset : presets ) {
      auto [rsz, asz] = default_grid_size(Z, RadialQuad::MuraKnowles, preset);
      REQUIRE( rsz > 0 );
      REQUIRE( asz > 0 );
    }

    for( auto rq : radial_quads ) {
      if( rq == RadialQuad::TreutlerAhlrichs && Z > 36 ) continue; // unsupported by design
      REQUIRE( default_radial_scaling_factor(rq, Z) > 0. );
    }
  }

  SECTION("create_default_unpruned_grid_spec builds a usable spec") {
    auto spec = MolecularGridDefaults<T>::create_default_unpruned_grid_spec(
      AtomicId(8), RadialQuad::MuraKnowles, AtomicGridSizeDefault::FineGrid );
    auto grid = SphericalGridFactory<T>::generate_grid(spec);
    REQUIRE( grid->npts() > 0 );
  }

  SECTION("make_atom_instances wires per-element templates to atom instances") {
    std::unordered_map<AtomicId, typename SphericalGridFactory<T>::spherical_grid_ptr> element_grids = {
      { 8, make_test_grid_ptr<T>(14, 50) },
      { 1, make_test_grid_ptr<T>(8, 26) },
    };
    std::vector<AtomicId> ids = { 8, 1, 1 };
    std::vector<cartesian_pt_t<T>> pos = { {T(0.),T(0.),T(0.)}, {T(1.),T(1.),T(0.)}, {T(-1.),T(1.),T(0.)} };

    auto atoms = make_atom_instances<T>(ids, pos, element_grids);
    REQUIRE( atoms.size() == 3 );
    REQUIRE( atoms[0].grid == element_grids.at(8) );
    REQUIRE( atoms[1].grid == element_grids.at(1) );
    REQUIRE( atoms[2].grid == element_grids.at(1) );

    REQUIRE_THROWS_AS(
      (make_atom_instances<T>({99}, {{T(0.),T(0.),T(0.)}}, element_grids)),
      std::out_of_range );
  }
}
