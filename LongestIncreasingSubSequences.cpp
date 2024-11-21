// Longest Increasing Sub-Sequences using Patience Sorting
// As per https://en.wikipedia.org/wiki/Patience_sorting

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <catch2/catch_all.hpp>

template <typename T>
struct Card
{
    T value;
    size_t sizeOfPreviousPile{}; // back-pointer to top of previous pile upon insertion
};

template <typename T>
using Pile = std::vector<Card<T>>;

template <typename T>
using Piles = std::vector<Pile<T>>;

template <typename T>
void print(const std::vector<T> &A, const char *msg = nullptr)
{
    if (msg)
    {
        std::cout << msg << ": ";
    }
    const char *prefix = "";
    std::cout << "{";
    std::for_each(A.begin(), A.end(), [&](const auto &a) { //
        std::cout << prefix << a;
        prefix = ", ";
    });
    std::cout << "}" << std::endl;
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
        std::cout << std::endl;
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
void checkPatienceProperties(const Piles<T> piles)
{
    // No pile can be empty
    assert(std::none_of(piles.begin(), piles.end(), [](const auto &p)
                        { return p.empty(); }));
    // The cards in each pile are in decreasing value order
    assert(std::all_of(piles.begin(), piles.end(), [](const auto &p)
                       { return std::is_sorted(p.begin(), p.end(), [](const auto &a, const auto &b)
                                               { return a.value > b.value; }); }));
    // The cards in each pile are in increasing sizeOfPreviousPile order
    assert(std::all_of(piles.begin(), piles.end(), [](const auto &p)
                       { return std::is_sorted(p.begin(), p.end(), [](const auto &a, const auto &b)
                                               { return a.sizeOfPreviousPile < b.sizeOfPreviousPile; }); }));
}

template <typename T>
Piles<T> patiencePiles(const std::vector<T> &values)
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
        else if (const auto it = std::lower_bound(piles.begin(), piles.end(), value, [](const auto &pile, const auto &value)
                                                  { return pile.back().value < value; });
                 it != piles.end())
        {
            // Found the first pile in the ordered pile vector whose top value is not less than the current value.
            // Push the current value on top of that pile, pointing to current top of the previous pile (if any)
            it->push_back({value, it == piles.begin() ? 0 : prev(it)->size()});
        }
    }
    // Check basic invariants
    checkPatienceProperties(piles);
    // Check total number of Cards is equal to the total number of values
    assert(values.size() == std::accumulate(piles.begin(), piles.end(), 0, [](auto acc, const auto &p)
                                            { return acc + p.size(); }));
    return piles;
}

template <typename T>
size_t longestIncreasingSubSequenceSize(const std::vector<T> &values)
{
    return patiencePiles(values).size();
}

template <typename T>
class LongestIncreasingSubSequencesGenerator
{
public:
    LongestIncreasingSubSequencesGenerator() = delete;
    LongestIncreasingSubSequencesGenerator(const std::vector<T> &values)
        : m_piles(strippedPiles(values))
    {
        m_positions.resize(m_piles.size());
        reset();
    }

    std::vector<T> get() const
    {
        std::vector<T> result(m_piles.size());
        for (size_t i = 0; i < m_piles.size(); ++i)
        {
            result[i] = m_piles[i][m_positions[i]].value;
        }
        return result;
    }

    bool next()
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
                applyConstraints(i);
                return true;
            }
            else if (pile.size() > 1)
            {
                position = topPosition(i - 1);
                applyConstraints(i);
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
    inline size_t topPosition(size_t i) const
    {
        return m_piles[i].size() - 1;
    }

    void applyConstraints(size_t n)
    {
        for (auto i = std::min(n, m_piles.size() - 1); i; --i)
        {
            // Skip values in the next pile that are smaller than the current value in the current pile
            const auto value = m_piles[i - 1][m_positions[i - 1]].value;
            const auto &nextPile = m_piles[i];
            auto &nextPosition = m_positions[i];
            while (nextPosition > 0 && nextPile[nextPosition].value < value)
            {
                --nextPosition;
            }
        }
    }

    static Piles<T> strippedPiles(const std::vector<T> &values)
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
            auto it = pile.begin();
            // TODO : optimise by a binary search on the decreasing values
            while (it != pile.end() && it->value > maxValue)
            {
                ++it;
            }
            if (const size_t offset = std::distance(pile.begin(), it); offset)
            {
                // TODO : instead use an array of min positions to avoid vector shifting overhead
                pile.erase(pile.begin(), it);
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

// As per https://codesays.com/2016/solution-to-slalom-skiing-by-codility/
int slalomSkiing(const std::vector<int> &A)
{
    using T = long long;
    T maxA = *(std::max_element(A.begin(), A.end())) + 1;
    std::vector<T> extended;
    for (auto i : A)
    {
        extended.push_back(2 * maxA + (T)i);
        extended.push_back(2 * maxA - (T)i);
        extended.push_back((T)i);
    }

    print(A, "Original");
    print(extended, "Extended");

    return longestIncreasingSubSequenceSize(extended);
}

// TEST CASES
//===========

TEST_CASE("Slalom Skiing")
{
    std::vector<int> A{15, 13, 5, 7, 4, 10, 12, 8, 2, 11, 5, 9, 3};
    CHECK(slalomSkiing(A) == 8);
}

TEST_CASE("LongestIncreasing 1")
{
    std::vector<int> A{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    CHECK(longestIncreasingSubSequenceSize(A) == 6);
}

TEST_CASE("LongestIncreasing 2")
{
    std::vector<int> A{3, 10, 2, 1, 20};
    CHECK(longestIncreasingSubSequenceSize(A) == 3);
}

TEST_CASE("LongestIncreasing 3")
{
    std::vector<int> A{10, 22, 9, 33, 21, 50, 41, 60, 80};
    CHECK(longestIncreasingSubSequenceSize(A) == 6);
}

TEST_CASE("LongestIncreasingSubSequences")
{
    std::vector<std::vector<int>> A = {
        {},
        {1},
        {1, 2},
        {2, 1},
        {3, 1, 4, 2},
        {3, 4, 1, 2},
        {5, 2, 2, 5},
        {10, 20, 3, 40},
        {8, 9, 12, 10, 11},
        {5, 4, 3, 2, 1},
        {3, 2, 6, 4, 5, 1},
        {2, 5, 3, 7, 11, 8, 10, 13, 6},
        {10, 22, 9, 33, 21, 50, 41, 60, 80},
        {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15},
    };

    for (const auto &a : A)
    {
        std::cout << std::endl;
        std::cout << "LONGEST SUB-SEQUENCES FOR ";
        print(a);
        LongestIncreasingSubSequencesGenerator g(a);
        do
        {
            print(g.get());
        } while (g.next());
    }
}

TEST_CASE("LongestIncreasingSubSequences on permutations")
{
    // This may take a while as we check every permutation
    std::vector<int> A{1, 2, 3, 4, 5, 6, 7, 8, 9};
    do
    {
        LongestIncreasingSubSequencesGenerator g(A);
        do
        {
            const auto solution = g.get();
            CHECK(std::is_sorted(solution.begin(), solution.end()));
        } while (g.next());
    } while (std::next_permutation(A.begin(), A.end()));
}
