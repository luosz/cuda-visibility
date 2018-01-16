#pragma once

#ifndef KMEANS_H

#include <iostream>
#include <stdio.h>
#include <arrayfire.h>
#include <af/util.h>
#include <cstdlib>

//using namespace af;

af::array distance(af::array data, af::array means)
{
	int n = data.dims(0); // Number of features
	int k = means.dims(1); // Number of means
	af::array data2 = tile(data, 1, k, 1);
	af::array means2 = tile(means, n, 1, 1);
	// Currently using manhattan distance
	// Can be replaced with other distance measures
	return sum(abs(data2 - means2), 2);
}
// Get cluster id of each location in data
af::array clusterize(const af::array data, const af::array means)
{
	// Get manhattan distance
	af::array dists = distance(data, means);
	// get the locations of minimum distance
	af::array idx, val;
	af::min(val, idx, dists, 1);
	// Return cluster IDs
	return idx;
}
af::array new_means(af::array data, af::array clusters, int k)
{
	int d = data.dims(2);
	af::array means = af::constant(0, 1, k, d);
	af::array clustersd = af::tile(clusters, 1, 1, d);
	gfor(af::seq ii, k) {
		means(af::span, ii, af::span) = af::sum(data * (clustersd == ii)) / (af::sum(clusters == ii) + 1e-5);
	}
	return means;
}
// kmeans(means, clusters, data, k)
// data:  input,  1D or 2D (range > [0-1])
// k:     input,  # desired means (k > 1)
// means: output, vector of means
void kmeans(af::array &means, af::array &clusters, const af::array in, int k, int iter = 100)
{
	unsigned n = in.dims(0); // Num features
	unsigned d = in.dims(2); // feature length
							 // reshape input
	af::array data = in * 0;
	// re-center and scale down data to [0, 1]
	af::array minimum = min(in);
	af::array maximum = max(in);
	gfor(af::seq ii, d) {
		data(af::span, af::span, ii) = (in(af::span, af::span, ii) - minimum(ii).scalar<float>()) / maximum(ii).scalar<float>();
	}
	// Initial guess of means
	means = af::randu(1, k, d);
	af::array curr_clusters = af::constant(0, data.dims(0)) - 1;
	af::array prev_clusters;
	// Stop updating after specified number of iterations
	for (int i = 0; i < iter; i++) {
		// Store previous cluster ids
		prev_clusters = curr_clusters;
		// Get cluster ids for current means
		curr_clusters = clusterize(data, means);
		// Break early if clusters not changing
		unsigned num_changed = af::count<unsigned>(prev_clusters != curr_clusters);
		if (num_changed < (n / 1000) + 1) break;
		// Update current means for new clusters
		means = new_means(data, curr_clusters, k);
	}
	// Scale up means
	gfor(af::seq ii, d) {
		means(af::span, af::span, ii) = maximum(ii) * means(af::span, af::span, ii) + minimum(ii);
	}
	clusters = prev_clusters;
}

#endif // KMEANS_H
