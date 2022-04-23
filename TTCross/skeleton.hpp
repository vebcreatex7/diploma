#include "matrix.hpp"
#include "calculation.hpp"
#include "unfoldingMatrix.hpp"
#include <bits/stdc++.h>
using namespace std;

tuple<vector<int>, vector<int>> Skeleton(TMatrix A, double eps);
tuple<vector<size_t>, vector<size_t>> Skeleton(UnfoldingMatrix A, double eps);