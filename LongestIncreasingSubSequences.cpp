// Longest Increasing Sub-Sequences using Patience Sorting
// As per https://en.wikipedia.org/wiki/Patience_sorting

#include <algorithm>
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

template <typename T>
class LongestIncreasingSubSequencesGenerator
{
public:
    LongestIncreasingSubSequencesGenerator() = delete;
    ~LongestIncreasingSubSequencesGenerator() = default;
    LongestIncreasingSubSequencesGenerator(const LongestIncreasingSubSequencesGenerator &) = delete;
    auto operator=(const LongestIncreasingSubSequencesGenerator &) -> LongestIncreasingSubSequencesGenerator & = delete;
    LongestIncreasingSubSequencesGenerator(LongestIncreasingSubSequencesGenerator &&) = delete;
    auto operator=(LongestIncreasingSubSequencesGenerator &&) -> LongestIncreasingSubSequencesGenerator & = delete;

    LongestIncreasingSubSequencesGenerator(const std::vector<T> &values)
        : m_piles(strippedPiles(values))
    {
        m_positions.resize(m_piles.size());
        reset();
    }

    [[nodiscard]] auto get() const -> std::vector<T>
    {
        std::vector<T> result(m_piles.size());
        for (size_t i = 0; i < m_piles.size(); ++i)
        {
            result[i] = m_piles[i][m_positions[i]].value;
        }
        return result;
    }

    [[nodiscard]] auto next() -> bool
    {
        while (countDown())
        {
            if (checkConstraints())
            {
                return true;
            }
        }
        return false;
    }

    void reset()
    {
        for (size_t i = 0; i < m_piles.size(); ++i)
        {
            m_positions[i] = topPosition(i);
        }
    }

private:
    [[nodiscard]] auto topPosition(size_t i) const -> size_t
    {
        return m_piles[i].size() - 1;
    }

    [[nodiscard]] auto countDown() -> bool
    {
        // Try to advance to next position on last pile
        // If no next position, reset to first position, and apply logic on previous pile
        // If reached first pile, return false
        for (auto i = m_piles.size(); i; --i)
        {
            const auto &pile = m_piles[i - 1];
            auto &position = m_positions[i - 1];
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

    [[nodiscard]] auto checkConstraints() const -> bool
    {
        if (m_piles.empty())
        {
            return true;
        }
        bool ok{true};
        for (auto i = m_piles.size() - 1; i && ok; --i)
        {
            const auto position = m_positions[i - 1];
            const auto value = m_piles[i - 1][position].value;
            const auto &nextCard = m_piles[i][m_positions[i]];
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
    std::vector<size_t> m_positions;
};

// Generate all bitsets with S 1’s out of N, 0 <= S <= N <= 16
class CombinationGenerator
{
public:
    CombinationGenerator(size_t N, size_t S)
    {
        assert(N <= 16); // At most 16 bits supported
        assert(S <= N);  // Number of set bits must be less than total number of bits
        m_firstVal = (1 << S) - 1;
        m_lastVal = m_firstVal << (N - S);
    }

    [[nodiscard]] auto first() const -> size_t { return m_firstVal; }
    [[nodiscard]] auto last() const -> size_t { return m_lastVal; }

    [[nodiscard]] static auto next(size_t v) -> size_t
    {
        auto t = (v | (v - 1)) + 1;
        return t | ((((t & -t) / (v & -v)) >> 1) - 1);
    }

private:
    size_t m_firstVal{};
    size_t m_lastVal{};
};

template <typename T>
class SlowLongestIncreasingSubSequencesGenerator
{
public:
    SlowLongestIncreasingSubSequencesGenerator(const std::vector<T> &values)
        : m_values{values}
    {
        ensureOrder();
    }

    [[nodiscard]] auto get() const -> std::vector<T>
    {
        std::vector<T> result(m_sequenceSize);
        for (size_t i = 0, j = 0; i < m_values.size(); ++i)
        {
            auto bit = 1 << i;
            if (m_currentVal & bit)
            {
                result[j++] = m_values[i];
            }
        }
        return result;
    }

    [[nodiscard]] auto next() -> bool
    {
        if (m_currentVal == m_generator.last())
        {
            return false;
        }
        m_currentVal = m_generator.next(m_currentVal);
        return ensureOrder();
    }

private:
    [[nodiscard]] auto checkOrder() const -> bool
    {
        bool ok{true};
        std::optional<int> prev;
        bool ordered{true};
        for (auto i = 0; i < m_values.size() && ordered; ++i)
        {
            auto bit = 1 << i;
            if (m_currentVal & bit)
            {
                const auto next = m_values[i];
                if (prev && prev > next)
                {
                    ordered = false;
                }
                else
                {
                    prev = next;
                }
            }
        }
        return ordered;
    }

    auto ensureOrder() -> bool
    {
        while (!checkOrder())
        {
            if (m_currentVal == m_generator.last())
            {
                return false;
            }
            m_currentVal = m_generator.next(m_currentVal);
        }
        return true;
    }

    std::vector<T> m_values;
    size_t m_sequenceSize{longestIncreasingSubSequenceSize(m_values)};
    CombinationGenerator m_generator{m_values.size(), m_sequenceSize};
    size_t m_currentVal{m_generator.first()};
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

#include <catch2/catch_all.hpp>

void time_it(std::function<void()> timed, const std::string &what = "EXEC TIME:")
{
    const auto start = std::chrono::system_clock::now();
    timed();
    const auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> time = stop - start;
    std::cout << std::setw(8) << time.count() << " for " << what << "\n";
}

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
    auto val = g.first();
    while (true)
    {
        ++count;
        if (val == g.last())
        {
            break;
        }
        val = g.next(val);
    }

    CHECK(count == combinationCount(N, S));
};

auto countIncreasingSubSequences(const std::vector<int> &values) -> size_t
{
    const size_t S = longestIncreasingSubSequenceSize(values);
    const size_t N = values.size();

    if (S == 1)
    {
        return N;
    }
    if (S == N)
    {
        return 1;
    }
    CombinationGenerator g{N, S};
    size_t count{};
    auto val = g.first();
    while (true)
    {
        std::optional<int> prev;
        bool ordered{true};
        for (size_t i = 0; i < N && ordered; ++i)
        {
            auto bit = 1 << i;
            if (val & bit)
            {
                const auto next = values[i];
                if (prev && prev > next)
                {
                    ordered = false;
                }
                else
                {
                    prev = next;
                }
            }
        }
        if (ordered)
        {
            ++count;
        }
        if (val == g.last())
        {
            break;
        }
        val = g.next(val);
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
        do
        {
        } while (generator.next());
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
        do
        {
            print(generator.get());
        } while (generator.next());
    }
}

TEST_CASE("LongestIncreasingSubSequences on permutations")
{
    // This may take a while as we check every permutation
    constexpr size_t N = 8;
    std::vector<int> values(N);
    std::iota(values.begin(), values.end(), 1);
    do
    {
        LongestIncreasingSubSequencesGenerator generator{values};
        do
        {
            const auto solution = generator.get();
            CHECK(std::is_sorted(solution.begin(), solution.end()));
        } while (generator.next());
    } while (std::next_permutation(values.begin(), values.end()));
}

TEST_CASE("CombinationGenerator")
{
    testCombinationGenerator(16, 3);
    testCombinationGenerator(16, 0);
    testCombinationGenerator(16, 16);
    testCombinationGenerator(0, 0);
}

TEST_CASE("countIncreasingSubSequences")
{
    CHECK(countIncreasingSubSequences({2, 4, 1, 3}) == 3);
    CHECK(countIncreasingSubSequences({3, 4, 1, 2}) == 2);
    CHECK(countIncreasingSubSequences({3, 10, 2, 1, 20}) == 1);
    CHECK(countIncreasingSubSequences({0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}) == 4);
}

TEST_CASE("LongestIncreasingSubSequences Count Checking")
{
    // This may take a while as we check every permutation
    constexpr size_t N = 8;
    std::vector<int> values(N);
    std::iota(values.begin(), values.end(), 1);
    do
    {
        LongestIncreasingSubSequencesGenerator generator{values};
        size_t lissCount{};
        do
        {
            ++lissCount;
        } while (generator.next());
        const auto checkCount = countIncreasingSubSequences(values);
        CHECK(lissCount == checkCount);
    } while (std::next_permutation(values.begin(), values.end()));
}

TEST_CASE("LongestIncreasingSubSequencesGenerator Performance")
{
    // This may take a while as we check every permutation
    std::cout << "Performance" << '\n';
    constexpr size_t N = 9;
    time_it([]()
            { permutationsTest<LongestIncreasingSubSequencesGenerator<int>>(N); }, "Fast");
    time_it([]()
            { permutationsTest<SlowLongestIncreasingSubSequencesGenerator<int>>(N); }, "Slow");
}
