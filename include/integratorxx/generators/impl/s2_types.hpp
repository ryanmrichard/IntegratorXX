#pragma once
#include <integratorxx/quadratures/s2.hpp>

namespace IntegratorXX {
namespace detail {

// Kept inside `detail` (rather than directly in `IntegratorXX`) so these
// generic aliases don't collide with the fixed-`double` `ah_type`/etc.
// aliases that consumers commonly declare for themselves.
template <typename T>
using ah_type = AhrensBeylkin<T>;
template <typename T>
using de_type = Delley<T>;
template <typename T>
using ll_type = LebedevLaikov<T>;
template <typename T>
using wo_type = Womersley<T>;

}
}
