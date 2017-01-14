#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;

int main()
{
        double k = 18;
        string s;
        while (getline(cin, s)) {
                istringstream is(s);
                vector<double> data;
                vector<double> avg;
                double p, sum = 0;
                while (is >> p)
                        data.push_back(p);
                for (int i = 0; i < k; ++i)
                        sum += data[i];
                double last = 0;
                for (int i = k; i < (int)data.size(); ++i) {
                        sum += data[k];
                        sum -= last;
                        last = data[i - k];
                        avg.push_back(sum / k);
                }
                for (int i = 1; i < (int)avg.size(); ++i) {
                        cout << avg[i] - avg[i - 1] << endl;
                }
        }
        return 0;
}
