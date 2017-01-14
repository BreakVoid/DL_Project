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
        string s;
        while (getline(cin, s)) {
                istringstream is(s);
                string a;
                int cnt = 0;
                while (is >> a) {
                        cnt ++;
                }
                cout << cnt;
                break;
        }
        return 0;

}
