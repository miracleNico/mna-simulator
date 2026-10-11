// Self-registration of binding units.
//
// Each bindings/bind_<unit>.cpp defines `void bind_<unit>(nanobind::module_&)`
// and adds `static const mna::bindings::Registrar reg_<unit>{"<unit>", &bind_<unit>};`
// module.cpp calls every registered function in registration order (which is
// the link order of the translation units; units must not depend on each other
// at import time beyond nanobind type lookups, which are resolved lazily).
#pragma once

#include <nanobind/nanobind.h>

#include <vector>

namespace mna::bindings {

using BindFn = void (*)(nanobind::module_&);

struct Entry {
    const char* name;
    BindFn fn;
};

/// The registry: a function-local static so it is initialized before first use
/// regardless of translation-unit initialization order.
inline std::vector<Entry>& registry() {
    static std::vector<Entry> entries;
    return entries;
}

struct Registrar {
    Registrar(const char* name, BindFn fn) { registry().push_back({name, fn}); }
};

}  // namespace mna::bindings
