// Longest Increasing Sub-Sequences using Patience Sorting
// As per https://en.wikipedia.org/wiki/Patience_sorting

#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
struct Card
{
    T value{};
    size_t sizeOfPreviousPile{}; // back-pointer to top of previous pile upon insertion
};

template <typename T>
using Pile = std::vector<Card<T>>;

template <typename T>
using Piles = std::vector<Pile<T>>;

template <typename T>
void print(const std::vector<T> &values, const char *msg = nullptr)
{
    if (msg)
    {
        std::cout << msg << ": ";
    }
    const char *prefix = "";
    std::cout << "{";
    std::for_each(values.begin(), values.end(), [&](const auto &value) { //
        std::cout << prefix << value;
        prefix = ", ";
    });
    std::cout << "}" << '\n';
}

template <typename T>
void print(const Piles<T> &piles)
{
    std::for_each(piles.begin(), piles.end(), [](const auto &pile) { //
        const char *prefix = "";
        std::for_each(pile.begin(), pile.end(), [&](const auto &card) { //
            std::cout << prefix << card.value << "-" << card.sizeOfPreviousPile;
            prefix = ", ";
        });
        std::cout << '\n';
    });
}

// Useful patiencePiles invariants and properties:
// 1. It is either empty (iff empty values), or consists of non-empty Piles
// 2. The Cards in each Pile are in decreasing value order
// 3. The Cards in each Pile are in increasing sizeOfPreviousPile order
// 4. The number of Piles is the length of every largest increasing sub-sequence of values
// 5. An largest increasing sub-sequence can be generated in reverse by following the sizeOfPreviousPile from the last pile down
// 6. The total number of Cards is equal to the total number of values
template <typename T>
void checkPatienceProperties(const Piles<T> &piles)
{
    // No pile can be empty
    assert(std::none_of(piles.begin(), piles.end(), [](const auto &pile)
                        { return pile.empty(); }));
    // The cards in each pile are in decreasing value order
    assert(std::all_of(piles.begin(), piles.end(), [](const auto &pile)
                       { return std::is_sorted(pile.begin(), pile.end(), [](const auto &lhs, const auto &rhs)
                                               { return lhs.value > rhs.value; }); }));
    // The cards in each pile are in increasing sizeOfPreviousPile order
    assert(std::all_of(piles.begin(), piles.end(), [](const auto &pile)
                       { return std::is_sorted(pile.begin(), pile.end(), [](const auto &lhs, const auto &rhs)
                                               { return lhs.sizeOfPreviousPile < rhs.sizeOfPreviousPile; }); }));
}

template <typename T>
auto patiencePiles(const std::vector<T> &values) -> Piles<T>
{
    Piles<T> piles;
    for (const auto &value : values)
    {
        if (piles.empty() || value > piles.back().back().value)
        {
            // Current value is greater than top value of the last pile
            // Push new pile with current value, pointing to current top of the previous pile (if any)
            piles.push_back({{value, piles.empty() ? 0 : piles.back().size()}});
        }
        else if (const auto found = std::lower_bound(piles.begin(), piles.end(), value, [](const auto &pile, const auto &value)
                                                     { return pile.back().value < value; });
                 found != piles.end())
        {
            // Found the first pile in the ordered pile vector whose top value is not less than the current value.
            // Push the current value on top of that pile, pointing to current top of the previous pile (if any)
            found->push_back({value, found == piles.begin() ? 0 : prev(found)->size()});
        }
    }
    // Check basic invariants
    checkPatienceProperties(piles);
    // Check total number of Cards is equal to the total number of values
    assert(values.size() == std::accumulate(piles.begin(), piles.end(), 0, [](auto acc, const auto &pile)
                                            { return acc + pile.size(); }));
    return piles;
}

template <typename T>
auto longestIncreasingSubSequenceSize(const std::vector<T> &values) -> size_t
{
    return patiencePiles(values).size();
}

// Forward Iterator that takes a Generator which must provide the following:
//      auto first() const -> Value (provide first value)
//      auto next(Value& value) const -> bool (advance value to next state, or return false if last state)
//      auto get(const Value& value) const -> Result (return result for given iterator value)
// For it to be useful, the Generator is also expected to expose the begin() and end() functions:
//      auto begin() const -> GeneratorIterator
//      auto end() const -> GeneratorIterator
template <typename Generator, typename Value, typename Result>
class GeneratorIterator
{
public:
    static auto begin(const Generator &g) -> GeneratorIterator
    {
        return {g, g.first()};
    }

    static auto end(const Generator &g) -> GeneratorIterator
    {
        return {g};
    }

    friend auto operator==(const GeneratorIterator &lhs, const GeneratorIterator &rhs) -> bool
    {
        return &lhs.m_generator == &rhs.m_generator && lhs.m_value == rhs.m_value;
    }

    friend auto operator!=(const GeneratorIterator &lhs, const GeneratorIterator &rhs) -> bool
    {
        return !(lhs == rhs);
    }

    auto operator++() -> GeneratorIterator &
    {
        if (m_value && !m_generator.next(*m_value))
        {
            m_value.reset();
        }
        return *this;
    }

    auto operator++(int) -> GeneratorIterator
    {
        auto t = *this;
        ++(*this);
        return t;
    }

    auto operator*() const -> Result
    {
        assert(m_value);
        return m_generator.get(*m_value);
    }

private:
    GeneratorIterator(const Generator &g) : m_generator(g) {}
    GeneratorIterator(const Generator &g, Value v) : m_generator(g), m_value(std::move(v)) {}

    const Generator &m_generator;
    std::optional<Value> m_value;
};

// Generate all longest increasing sub-sequences using patiencePiles
template <typename T>
class LongestIncreasingSubSequencesGenerator
{
    using Positions = std::vector<size_t>;
    using Sequence = std::vector<T>;
    using Iterator = GeneratorIterator<LongestIncreasingSubSequencesGenerator, Positions, Sequence>;

public:
    LongestIncreasingSubSequencesGenerator() = delete;

    LongestIncreasingSubSequencesGenerator(const Sequence &values)
        : m_piles(strippedPiles(values))
    {
    }

    [[nodiscard]] auto begin() const -> Iterator { return Iterator::begin(*this); }
    [[nodiscard]] auto end() const -> Iterator { return Iterator::end(*this); }

private:
    friend Iterator;

    [[nodiscard]] auto get(const Positions &positions) const -> Sequence
    {
        Sequence result(m_piles.size());
        for (size_t i = 0; i < m_piles.size(); ++i)
        {
            result[i] = m_piles[i][positions[i]].value;
        }
        return result;
    }

    [[nodiscard]] auto first() const -> Positions
    {
        Positions positions(m_piles.size());
        for (size_t i = 0; i < m_piles.size(); ++i)
        {
            positions[i] = topPosition(i);
        }
        return positions;
    }

    [[nodiscard]] auto next(Positions &positions) const -> bool
    {
        while (countDown(positions))
        {
            if (checkConstraints(positions))
            {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] auto topPosition(size_t i) const -> size_t
    {
        return m_piles[i].size() - 1;
    }

    [[nodiscard]] auto countDown(Positions &positions) const -> bool
    {
        // Try to advance to next position on last pile
        // If no next position, reset to first position, and apply logic on previous pile
        // If reached first pile, return false
        for (auto i = m_piles.size(); i; --i)
        {
            const auto &pile = m_piles[i - 1];
            auto &position = positions[i - 1];
            if (position > 0)
            {
                --position;
                return true;
            }
            if (pile.size() > 1)
            {
                position = topPosition(i - 1);
            }
        }
        return false;
    }

    [[nodiscard]] auto checkConstraints(const Positions &positions) const -> bool
    {
        if (m_piles.empty())
        {
            return true;
        }
        bool ok{true};
        for (auto i = m_piles.size() - 1; i && ok; --i)
        {
            const auto position = positions[i - 1];
            const auto value = m_piles[i - 1][position].value;
            const auto &nextCard = m_piles[i][positions[i]];
            // Current value in the next pile must be larger than the current value in the current pile
            // Current value in the next pile must have been added after the current value in the current pile
            ok = nextCard.value > value && nextCard.sizeOfPreviousPile > position;
        }
        return ok;
    }

    [[nodiscard]] static auto strippedPiles(const std::vector<T> &values) -> Piles<T>
    {
        auto piles = patiencePiles(values);
        if (piles.empty())
        {
            return piles;
        }
        for (auto i = piles.size() - 1; i; --i)
        {
            auto &pile = piles[i - 1];
            const auto &nextPile = piles[i];
            // Remove cards at top of each pile that are not reachable from the next pile (via sizeOfPreviousPile)
            // (We use patiemcePiles invariants 1 and 3 here, so only need to consider last card)
            if (const auto maxSize = nextPile[nextPile.size() - 1].sizeOfPreviousPile; pile.size() > maxSize)
            {
                pile.resize(maxSize);
            }
            // Remove cards at bottom of each pile that are larger than the largest value of the next pile
            // (We use patiemcePiles invariants 1 and 2 here, so can take largest from first card)
            const auto maxValue = nextPile[0].value;
            auto iter = pile.begin();
            // TODO : optimise by a binary search on the decreasing values
            while (iter != pile.end() && iter->value > maxValue)
            {
                ++iter;
            }
            if (const size_t offset = std::distance(pile.begin(), iter); offset)
            {
                // TODO : instead use an array of min positions to avoid vector shifting overhead
                pile.erase(pile.begin(), iter);
                for (auto &card : piles[i])
                {
                    card.sizeOfPreviousPile -= offset;
                }
            }
        }
        // By construction, strippedPiles verifies all the invariants of patiencePiles, except the last
        checkPatienceProperties(piles);
        return piles;
    }

    const Piles<T> m_piles;
};

// Generate all bitsets with S 1’s out of N, 0 <= S <= N <= sizeof(T) * 8
template <typename T = u_int32_t>
class CombinationGenerator
{
public:
    using Iterator = GeneratorIterator<CombinationGenerator, T, T>;

    CombinationGenerator(size_t N, size_t S)
    {
        assert(N <= sizeof(T) * 8); // At most 8*bytecount bits supported
        assert(S <= N);             // Number of set bits must be less than total number of bits
        m_firstVal = (1 << S) - 1;
        m_lastVal = m_firstVal << (N - S);
    }

    [[nodiscard]] auto begin() const -> Iterator { return Iterator::begin(*this); }
    [[nodiscard]] auto end() const -> Iterator { return Iterator::end(*this); }

private:
    friend Iterator;

    [[nodiscard]] auto get(T val) const -> T
    {
        return val;
    }

    [[nodiscard]] auto first() const -> T
    {
        return m_firstVal;
    }

    [[nodiscard]] auto next(T &val) const -> bool
    {
        if (val != m_lastVal)
        {
            val = nextValue(val);
            return true;
        }
        return false;
    }

    [[nodiscard]] auto nextValue(T val) const -> T
    {
        // Taken from https://math.stackexchange.com/a/2255452
        auto c = val & -val;
        auto r = val + c;
        // c == 0 can only happen for val == 0, i.e. no bit set
        // This implies S == 0, and m_firstVal == m_lastVal == 0,
        // in which case nextValue() should never be called
        assert(c);
        return (((r ^ val) >> 2) / c) | r;
    }

    T m_firstVal{};
    T m_lastVal{};
};

// Check f(pos) on each set bit
template <typename F, typename T = u_int32_t>
auto all_of_set_bits(T val, F f) -> bool
{
    for (size_t pos = 0; val; ++pos, val >>= 1)
    {
        if ((val & 1) && !f(pos))
        {
            return false;
        }
    }
    return true;
}

// Apply f(pos) on each set bit
template <typename F, typename T = u_int32_t>
void for_each_set_bit(T val, F f)
{
    all_of_set_bits(val, [&f](auto pos)
                    {f(pos); return true; });
}

// Reverse order of bits
// Taken from https://stackoverflow.com/a/69171685
// Probably not the fastest, but generic and easy enough to understand
template <typename T = u_int32_t>
auto reverse_bits(T n) -> T
{
    // we force the passed-in type to its unsigned equivalent, because C++ may
    // perform arithmetic right shift instead of logical right shift, depending
    // on the compiler implementation.
    using unsigned_T = typename std::make_unsigned_t<T>;
    auto v = (unsigned_T)n;

    // swap every bit with its neighbor
    v = ((v & 0xAAAAAAAAAAAAAAAA) >> 1) | ((v & 0x5555555555555555) << 1);

    // swap every pair of bits
    v = ((v & 0xCCCCCCCCCCCCCCCC) >> 2) | ((v & 0x3333333333333333) << 2);

    // swap every nibble
    v = ((v & 0xF0F0F0F0F0F0F0F0) >> 4) | ((v & 0x0F0F0F0F0F0F0F0F) << 4);
    // bail out if we've covered the word size already
    if (sizeof(T) == 1)
        return v;

    // swap every byte
    v = ((v & 0xFF00FF00FF00FF00) >> 8) | ((v & 0x00FF00FF00FF00FF) << 8);
    if (sizeof(T) == 2)
        return v;

    // etc...
    v = ((v & 0xFFFF0000FFFF0000) >> 16) | ((v & 0x0000FFFF0000FFFF) << 16);
    if (sizeof(T) <= 4)
        return v;

    v = ((v & 0xFFFFFFFF00000000) >> 32) | ((v & 0x00000000FFFFFFFF) << 32);

    return v;
}

// SlowLongestIncreasingSubSequencesGenerator uses an CombinationGenerator to generate all combinations of
// S = longestIncreasingSubSequenceSize(values) set bits within N = values.size() bits
// Each combination maps to a potential longest increasing sub-sequence
// (where the positions of 1's in the combination are the positions of values in the provided vector)
// But we need to check each of these potential solutions and only return the increasing ones
// Hence this is much slower than the "stripped piles" version above, but gives more confidence that
// all potential solutions are covered.
// Checking that both generators give the same results on a large test set increases our confidence
// in the fast version.
template <typename T, typename U = u_int32_t>
class SlowLongestIncreasingSubSequencesGenerator
{
    using CombinationGenerator = CombinationGenerator<U>;
    using CombinationIterator = typename CombinationGenerator::Iterator;
    using Sequence = std::vector<T>;
    using Iterator = GeneratorIterator<SlowLongestIncreasingSubSequencesGenerator, CombinationIterator, Sequence>;

public:
    SlowLongestIncreasingSubSequencesGenerator(Sequence values)
        : m_values{std::move(values)}
    {
    }

    [[nodiscard]] auto begin() const -> Iterator { return Iterator::begin(*this); }
    [[nodiscard]] auto end() const -> Iterator { return Iterator::end(*this); }

private:
    friend Iterator;

    [[nodiscard]] auto get(const CombinationIterator &iterator) const -> std::vector<T>
    {
        assert(iterator != m_generator.end());
        std::vector<T> result(m_sequenceSize);
        size_t i = 0;
        for_each_set_bit(reverseVal(*iterator), [&](auto pos)
                         { result[i++] = m_values[pos]; });
        return result;
    }

    [[nodiscard]] auto first() const -> CombinationIterator
    {
        auto iterator = m_generator.begin();
        assert(ensureOrder(iterator));
        return iterator;
    }

    [[nodiscard]] auto next(CombinationIterator &iterator) const -> bool
    {
        if (iterator == m_generator.end())
        {
            return false;
        }
        ++iterator;
        return ensureOrder(iterator);
    }

    [[nodiscard]] auto checkOrder(const CombinationIterator &iterator) const -> bool
    {
        if (iterator == m_generator.end())
        {
            return false;
        }
        std::optional<int> prev;
        return all_of_set_bits(reverseVal(*iterator), [&](auto pos) { //
            const auto &next = m_values[pos];
            if (prev && prev > next)
            {
                return false;
            }
            prev = next;
            return true;
        });
    }

    [[nodiscard]] auto ensureOrder(CombinationIterator &iterator) const -> bool
    {
        while (!checkOrder(iterator))
        {
            if (iterator == m_generator.end())
            {
                return false;
            }
            ++iterator;
        }
        return true;
    }

    // This is used to make sure the solutions are returned in the same order as
    // the LongestIncreasingSubSequencesGenerator we're comparing with
    [[nodiscard]] auto reverseVal(U val) const -> U
    {
#if USE_BITSET
        std::bitset<32> bits{val};
        const auto len = m_values.size();
        for (size_t i = 0; i < len / 2; ++i)
        {
            const auto pos = i;
            const auto rpos = len - i - 1;
            const bool t = bits[pos];
            bits[pos] = bits[rpos];
            bits[rpos] = t;
        }
        return bits.to_ulong();
#else
        const auto reverse = reverse_bits<U>(val);
        return reverse >> (32 - m_values.size());
#endif
    }

    std::vector<T> m_values;
    size_t m_sequenceSize{longestIncreasingSubSequenceSize(m_values)};
    CombinationGenerator m_generator{m_values.size(), m_sequenceSize};
};

// As per https://codesays.com/2016/solution-to-slalom-skiing-by-codility/
auto slalomSkiing(const std::vector<int> &values) -> size_t
{
    using T = long long;
    T maxA = *(std::max_element(values.begin(), values.end())) + 1;
    std::vector<T> extended;
    for (auto value : values)
    {
        extended.push_back((2 * maxA) + (T)value);
        extended.push_back((2 * maxA) - (T)value);
        extended.push_back((T)value);
    }

    print(values, "Original");
    print(extended, "Extended");

    return longestIncreasingSubSequenceSize(extended);
}

// TEST CASES
//===========

#include <chrono>
#include <functional>
#include <iomanip>
#include <utility>

#include <catch2/catch_all.hpp>

// Measure execution time of a function
void time_it(const std::function<void()> &timed, const std::string &what = "EXEC TIME:")
{
    const auto start = std::chrono::system_clock::now();
    timed();
    const auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> time = stop - start;
    std::cout << std::setw(8) << time.count() << " for " << what << "\n";
}

// One of many nCk algorithms that avoids huge factorials...
auto combinationCount(size_t n, size_t k) -> size_t
{
    if (k > n)
    {
        return 0;
    }
    k = std::min(k, n - k);
    if (k == 0)
    {
        return 1;
    }
    size_t result = n;
    for (size_t i = 2; i <= k; ++i)
    {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

void testCombinationGenerator(size_t N, size_t S)
{
    CombinationGenerator g{N, S};
    size_t count{};
    for (auto val : g)
    {
        size_t nbSet{};
        for_each_set_bit(val, [&nbSet](auto)
                         { ++nbSet; });
        // Each combination must have exactly S bits set
        CHECK(nbSet == S);
        ++count;
    }
    // The total number of combinations must match nCk
    CHECK(count == combinationCount(N, S));
};

// Brute force going through all candidate combinations and counting ordered ones
auto countIncreasingSubSequences(const std::vector<int> &values) -> size_t
{
    const size_t S = longestIncreasingSubSequenceSize(values);
    const size_t N = values.size();
    CombinationGenerator g{N, S};
    size_t count{};
    for (auto val : g)
    {
        std::optional<int> prev;
        const bool ordered = all_of_set_bits(val, [&](auto pos) { //
            const auto &next = values[pos];
            if (prev && prev > next)
            {
                return false;
            }
            prev = next;
            return true;
        });

        if (ordered)
        {
            ++count;
        }
    }
    return count;
}

template <typename Generator>
void permutationsTest(size_t N)
{
    std::vector<int> values(N);
    std::iota(values.begin(), values.end(), 1);
    do
    {
        Generator generator{values};
        for (auto solution : generator)
        {
            CHECK(std::is_sorted(solution.begin(), solution.end()));
        }
    } while (std::next_permutation(values.begin(), values.end()));
}

TEST_CASE("Slalom Skiing")
{
    const std::vector<int> values{15, 13, 5, 7, 4, 10, 12, 8, 2, 11, 5, 9, 3};
    CHECK(slalomSkiing(values) == 8);
}

TEST_CASE("LongestIncreasingSubSequenceSize")
{
    CHECK(longestIncreasingSubSequenceSize<int>({0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}) == 6);
    CHECK(longestIncreasingSubSequenceSize<int>({3, 10, 2, 1, 20}) == 3);
    CHECK(longestIncreasingSubSequenceSize<int>({10, 22, 9, 33, 21, 50, 41, 60, 80}) == 6);
}

TEST_CASE("LongestIncreasingSubSequences")
{
    std::vector<std::vector<int>> values = {
        {},
        {1},
        {1, 2},
        {2, 1},
        {2, 4, 1, 3},
        {3, 1, 4, 2},
        {3, 4, 1, 2},
        {5, 2, 2, 5},
        {10, 20, 3, 40},
        {3, 4, 1, 2, 5},
        {8, 9, 12, 10, 11},
        {5, 4, 3, 2, 1},
        {3, 2, 6, 4, 5, 1},
        {2, 5, 3, 7, 11, 8, 10, 13, 6},
        {10, 22, 9, 33, 21, 50, 41, 60, 80},
        {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15},
    };

    for (const auto &value : values)
    {
        std::cout << '\n';
        std::cout << "LONGEST SUB-SEQUENCES FOR ";
        print(value);
        LongestIncreasingSubSequencesGenerator generator{value};
        for (auto sequence : generator)
        {
            print(sequence);
        }
    }
}

TEST_CASE("CombinationGenerator")
{
    testCombinationGenerator(32, 3);
    testCombinationGenerator(16, 0);
    testCombinationGenerator(16, 16);
    testCombinationGenerator(0, 0);
}

TEST_CASE("countIncreasingSubSequences")
{
    CHECK(countIncreasingSubSequences({1, 2, 3, 4}) == 1);
    CHECK(countIncreasingSubSequences({4, 3, 2, 1}) == 4);
    CHECK(countIncreasingSubSequences({2, 4, 1, 3}) == 3);
    CHECK(countIncreasingSubSequences({3, 4, 1, 2}) == 2);
    CHECK(countIncreasingSubSequences({3, 10, 2, 1, 20}) == 1);
    CHECK(countIncreasingSubSequences({0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}) == 4);
}

TEST_CASE("LongestIncreasingSubSequences Value Checking")
{
    // This may take a while as we check every permutation
    constexpr size_t N = 8;
    std::vector<int> values(N);
    std::iota(values.begin(), values.end(), 1);
    do
    {
        LongestIncreasingSubSequencesGenerator fast{values};
        SlowLongestIncreasingSubSequencesGenerator slow{values};
        for (auto [itFast, itSlow] = std::pair{fast.begin(), slow.begin()};
             itFast != fast.end() && itSlow != slow.end();
             ++itFast, ++itSlow)
        {
            const auto solution = *itFast;
            CHECK(std::is_sorted(solution.begin(), solution.end()));
            CHECK(solution == *itSlow);
        }
    } while (std::next_permutation(values.begin(), values.end()));
}

TEST_CASE("LongestIncreasingSubSequences Count Checking")
{
    // This may take a while as we check every permutation
    constexpr size_t N = 8;
    std::vector<int> values(N);
    std::iota(values.begin(), values.end(), 1);
    do
    {
        LongestIncreasingSubSequencesGenerator fast{values};
        SlowLongestIncreasingSubSequencesGenerator slow{values};
        size_t fastCount{};
        size_t slowCount{};
        std::for_each(fast.begin(), fast.end(), [&fastCount](auto)
                      { ++fastCount; });
        std::for_each(slow.begin(), slow.end(), [&slowCount](auto)
                      { ++slowCount; });
        const auto checkCount = countIncreasingSubSequences(values);
        CHECK(fastCount == checkCount);
        CHECK(slowCount == checkCount);
    } while (std::next_permutation(values.begin(), values.end()));
}

TEST_CASE("LongestIncreasingSubSequencesGenerator Performance")
{
    // This may take a while as we check every permutation
    std::cout << "Performance" << '\n';
    constexpr size_t N = 8;
    time_it([]()
            { permutationsTest<LongestIncreasingSubSequencesGenerator<int>>(N); }, "Fast");
    time_it([]()
            { permutationsTest<SlowLongestIncreasingSubSequencesGenerator<int>>(N); }, "Slow");
}
