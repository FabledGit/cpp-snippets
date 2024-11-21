cmake_minimum_required( VERSION 3.10 )

project( CPPSnippets LANGUAGES CXX )

set (CMAKE_CXX_STANDARD 17)

find_package(Catch2 3 REQUIRED)

set(SOURCES
        LongestIncreasingSubSequences.cpp
)

add_executable( cpp-snippets ${SOURCES} )

target_link_libraries( cpp-snippets Catch2::Catch2WithMain )
