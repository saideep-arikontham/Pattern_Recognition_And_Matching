#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

float earth_movers_distance(vector<float> &img_data1, vector<float> &img_data2) {
    if (img_data1.size() != img_data2.size()) {
        cerr << "Histograms must have the same number of bins." << endl;
        return -1.0;
    }

    float emd = 0.0;
    float cumulative_flow = 0.0;  // Accumulate mass flow from one bin to the next

    for (size_t i = 0; i < img_data1.size(); i++) {
        cumulative_flow += img_data1[i] - img_data2[i];  // Compute flow
        emd += abs(cumulative_flow);  // Accumulate absolute work required
    }

    return emd;
}

int main() {
    vector<float> hist1 = {0.1, 0.4, 0.2, 0.3};  // Example histogram 1
    vector<float> hist2 = {0.1, 0.4, 0.2, 0.3};  // Example histogram 2

    float distance = earth_movers_distance(hist1, hist2);
    cout << "Earth Mover's Distance: " << distance << endl;

    return 0;
}
