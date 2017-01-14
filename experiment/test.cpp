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
        string s;
        while (getline(cin, s)) {
                istringstream is(s);
                vector<double> data;
                double p, sum;
                while (is >> p) {
                        data.push_back(p);
                        sum += p;
                }
                sum /= (double)data.size();
                for (vector<double>::iterator i = data.begin(); i < data.end(); ++i) {
                        cout << (*i) - sum << endl;
                }
        }
        return 0;
}
