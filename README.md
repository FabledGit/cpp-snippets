# C++ Snippets

This repository contains C++ solutions to some interesting problems, such as [Codility](https://app.codility.com/programmers/trainings/) exercises


They use the [Catch2](https://github.com/catchorg/Catch2) test framework for unit tests, which can also be installed via [Homebrew](https://formulae.brew.sh/formula/catch2)

## Longest increasing sub-sequences
 I started exploring algorithms to generate the longest increasing sub-sequences of a sequence of values while solving the [SlalomSkiing](https://app.codility.com/programmers/trainings/1/slalom_skiing/) codility exercise. The latter only requires computing the length of any such sub-sequence. This is achieved in O(nlog(n)) time via a Patience Piles construction algorithm. I went one step further to provide a class that generates all such sub-sequences.

## Type Erasure
Type-erased shapes example, based on Klaus Iglberger's [talk](https://www.youtube.com/watch?v=jKt6A3wnDyI) at Meeting Cpp 2022. This shows how to implement a collection of independent shapes along with their independent and configurable drawing strategies, without using the dreaded shape inheritance design pattern.
