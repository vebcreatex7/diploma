#ifndef SKELETON_H
#define SKELETON_H

#include <bits/stdc++.h>

#include "matrix.hpp"
#include "calculation.hpp"
#include "unfolding_matrix.hpp"

using namespace std;

tuple<vector<int>, vector<int>> Skeleton(TMatrix A, double eps);
std::pair<std::vector<size_t>,std::vector<size_t>> Skeleton(UnfoldingMatrix A, size_t maxR, double eps);
std::pair<std::vector<size_t>,std::vector<size_t>> SlowSkeleton(UnfoldingMatrix A, size_t maxR, double eps);

#endif