#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

class vec_3 
{
    public:
        float x;
        float y;
        float z;
        
        vec_3(){}

        vec_3(float x1, float y1, float z1)
        {
            x = x1;
            y = y1;
            z = z1;
        }
        
        vec_3 operator+(const vec_3& vec_a)
        {
            
            vec_3 sum;
            sum.x = this->x + vec_a.x;
            sum.y = this->y + vec_a.y;
            sum.z = this->z + vec_a.z;
            return sum;
        }

        vec_3 operator-(const vec_3& vec_a)
        {
            
            vec_3 sum;
            sum.x = this->x - vec_a.x;
            sum.y = this->y - vec_a.y;
            sum.z = this->z - vec_a.z;
            return sum;
        }
        
};

float norm(vec_3& vec)
{
    float normed;
    normed = pow(vec.x,2)+pow(vec.y,2)+pow(vec.z,2);
    normed = sqrt(normed);
    return normed;
}

ostream& operator<<(ostream& os, const vec_3& vec)
{
    os << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
    return os;
}

class grav_obj
{
    public:
        float mass;
        vec_3 pos;
        vec_3 vel;
        vec_3 half_vel;
        vec_3 accel_buffer;
        
};

void integrate_system(vector<grav_obj> &sys)
{
    int n = sys.size();
    for (int i=0; i < n; i++)
    {
        for (int i=0; i < n; i++)
        {
            
        }
    }
}

int factorial(int k)
{
    int fact = 1;
    for (int i=1; i < k+1; i++){
        fact = fact*i;
    }
    return fact;
}

int main()
{
    vec_3 A(0,2,3);
    vec_3 B(-5,1,5);
    cout << A.y << endl;
    cout << A+B << endl;
    cout << factorial(5) << endl;
}