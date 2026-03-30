# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/zjj/vllm/build-clangd/_deps/onednn-src")
  file(MAKE_DIRECTORY "/home/zjj/vllm/build-clangd/_deps/onednn-src")
endif()
file(MAKE_DIRECTORY
  "/home/zjj/vllm/build-clangd/_deps/onednn-build"
  "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix"
  "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/tmp"
  "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/src/onednn-populate-stamp"
  "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/src"
  "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/src/onednn-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/src/onednn-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/zjj/vllm/build-clangd/_deps/onednn-subbuild/onednn-populate-prefix/src/onednn-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
