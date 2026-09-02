#pragma once
#include <integratorxx/quadratures/radial.hpp>

namespace IntegratorXX {
namespace detail {

// Kept inside `detail` (rather than directly in `IntegratorXX`) so these
// generic aliases don't collide with the fixed-`double` `bk_type`/etc.
// aliases that consumers commonly declare for themselves.
template <typename T>
using bk_type  = Becke<T,T>;
template <typename T>
using mk_type  = MuraKnowles<T,T>;
template <typename T>
using mhl_type = MurrayHandyLaming<T,T>;
template <typename T>
using ta_type  = TreutlerAhlrichs<T,T>;

}
}

