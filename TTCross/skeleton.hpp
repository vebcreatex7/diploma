#ifndef SKELETON_H
#define SKELETON_H

#include "matrix.hpp"
#include "calculation.hpp"
#include "unfolding_matrix.hpp"
#include <bits/stdc++.h>
using namespace std;

tuple<vector<int>, vector<int>> Skeleton(TMatrix A, double eps);
std::pair<std::vector<size_t>,std::vector<size_t>> Skeleton(UnfoldingMatrix A, size_t maxR, double eps);

#endif