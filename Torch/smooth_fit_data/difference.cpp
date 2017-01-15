#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>

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
                for (int i = 20; i < 100; ++i) {
                        cout << data[i] << endl;
                }
        }
        return 0;
}
