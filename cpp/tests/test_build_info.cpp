#include <doctest/doctest.h>

#include <string>

#include "mna/version.hpp"

TEST_CASE("version and build info") {
    CHECK(std::string(mna::version()) == "0.70.0-dev");
    CHECK(std::string(mna::version()) == mna::kVersionString);

    const mna::BuildInfo info = mna::build_info();
    CHECK(info.version == mna::kVersionString);
    CHECK_FALSE(info.compiler.empty());
    CHECK_FALSE(info.compiler_version.empty());
    CHECK(info.cxx_standard >= 201703L);
    CHECK_FALSE(info.build_type.empty());
    CHECK(info.threads_impl == "std::thread");
    CHECK(info.eigen_version == "3.4.0");
    CHECK(info.suitesparse_version == "7.8.3");
    CHECK(info.klu_version == "2.3.5");
    CHECK_FALSE(info.platform.empty());
}
