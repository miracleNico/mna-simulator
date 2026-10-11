// Example self-registered binding unit; also exposes the list of registered units.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include "registry.hpp"

namespace nb = nanobind;

namespace {

void bind_meta(nb::module_& m) {
    m.def(
        "binding_units",
        [] {
            std::vector<std::string> names;
            for (const mna::bindings::Entry& entry : mna::bindings::registry()) {
                names.emplace_back(entry.name);
            }
            return names;
        },
        "Names of the self-registered binding units compiled into _core, in registration order.");
}

const mna::bindings::Registrar reg_meta{"meta", &bind_meta};

}  // namespace
