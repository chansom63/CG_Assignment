#include <SDL2/SDL.h>
#include <cmath>
#include <vector>
#include <algorithm> // Needed for std::clamp, std::max, std::min

struct Vec3
{
    double x, y, z;
    Vec3(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    Vec3 operator*(double b) const { return Vec3(x * b, y * b, z * b); }
    Vec3 operator/(double b) const { return Vec3(x / b, y / b, z / b); }
    double dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    Vec3 cross(const Vec3 &b) const
    {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }
    Vec3 normalize() const
    {
        double l = sqrt(x * x + y * y + z * z);
        return (l > 0) ? *this / l : *this;
    }
};

struct Ray
{
    Vec3 orig, dir;
    Ray(Vec3 o, Vec3 d) : orig(o), dir(d) {}
};

struct Triangle
{
    Vec3 v0, v1, v2, color;
    Triangle(Vec3 a, Vec3 b, Vec3 c, Vec3 col) : v0(a), v1(b), v2(c), color(col) {}
};

bool intersect(const Ray &ray, const Triangle &tri, double &t, Vec3 &normal)
{
    const double EPS = 1e-6;
    Vec3 e1 = tri.v1 - tri.v0;
    Vec3 e2 = tri.v2 - tri.v0;
    Vec3 h = ray.dir.cross(e2);
    double a = e1.dot(h);
    if (fabs(a) < EPS)
        return false;
    double f = 1.0 / a;
    Vec3 s = ray.orig - tri.v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
        return false;
    Vec3 q = s.cross(e1);
    double v = f * ray.dir.dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    t = f * e2.dot(q);
    if (t > EPS)
    {
        normal = e1.cross(e2).normalize();
        return true;
    }
    return false;
}

Vec3 trace(const Ray &ray, const std::vector<Triangle> &tris, const Vec3 &lightPos, int depth = 0)
{
    double nearest = 1e9;
    const Triangle *hitTri = nullptr;
    Vec3 hitNormal;

    for (auto &tri : tris)
    {
        double t;
        Vec3 n;
        if (intersect(ray, tri, t, n) && t < nearest)
        {
            nearest = t;
            hitTri = &tri;
            hitNormal = n;
        }
    }

    if (!hitTri)
        return Vec3(0.5, 0.7, 1.0); // sky background

    Vec3 hitPoint = ray.orig + ray.dir * nearest;
    Vec3 lightDir = (lightPos - hitPoint).normalize();

    // Diffuse + ambient lighting
    double diff = std::max(0.0, hitNormal.dot(lightDir));
    Vec3 ambient = hitTri->color * 0.3;
    Vec3 diffuse = hitTri->color * diff * 0.7;
    Vec3 color = ambient + diffuse;

    // Simple reflection
    if (depth < 1)
    {
        Vec3 reflDir = ray.dir - hitNormal * 2.0 * ray.dir.dot(hitNormal);
        Vec3 reflColor = trace(Ray(hitPoint + hitNormal * 1e-4, reflDir.normalize()), tris, lightPos, depth + 1);
        color = color * 0.8 + reflColor * 0.2;
    }

    // ----------------------------------------
    // **MODIFICATION: Facing-ratio brightness adjustment**
    // ----------------------------------------
    // Calculate the dot product between the view direction and the surface normal.
    // A value of 1 means the face is pointing directly at the camera.
    // A value of 0 means the face is perpendicular to the camera view.
    double facingRatio = std::max(0.0, hitNormal.dot(ray.dir * -1.0));

    // Use this ratio to create a brightness factor.
    // 0.6 is a base brightness, and we add up to 0.9 based on the facing angle.
    // This makes faces directly in front (facingRatio ≈ 1) much brighter.
    double brightnessFactor = 0.6 + 0.9 * facingRatio;
    color = color * brightnessFactor;
    // ----------------------------------------

    // Clamp to [0, 1]
    color.x = std::min(1.0, std::max(0.0, color.x));
    color.y = std::min(1.0, std::max(0.0, color.y));
    color.z = std::min(1.0, std::max(0.0, color.z));
    return color;
}

std::vector<Triangle> makeCube()
{
    std::vector<Triangle> tris;
    Vec3 green(0, 1, 0);
    double s = 1.0;
    Vec3 v[8] = {
        {-s, -s, -s}, {s, -s, -s}, {s, s, -s}, {-s, s, -s}, {-s, -s, s}, {s, -s, s}, {s, s, s}, {-s, s, s}};
    int faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4}, {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {0, 3, 7}, {0, 7, 4}};
    for (auto &f : faces)
        tris.emplace_back(v[f[0]], v[f[1]], v[f[2]], green);
    return tris;
}

int main()
{
    const int width = 400, height = 400;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Ray Traced Cube (Facing Ratio Brightness)",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          width, height, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);

    std::vector<Triangle> tris = makeCube();
    Vec3 lightPos(3, 3, 3);

    bool quit = false;
    SDL_Event e;
    double angle = 0.0;

    while (!quit)
    {
        while (SDL_PollEvent(&e))
            if (e.type == SDL_QUIT)
                quit = true;

        angle += 0.02;
        double camX = 4.0 * cos(angle), camZ = 4.0 * sin(angle);
        Vec3 camPos(camX, 1.5, camZ);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double u = (2.0 * (x + 0.5) / width - 1.0) * (width / (double)height);
                double v = (1.0 - 2.0 * (y + 0.5) / height);
                Vec3 forward = (Vec3(0, 0, 0) - camPos).normalize();
                Vec3 right = Vec3(0, 1, 0).cross(forward).normalize();
                Vec3 up = forward.cross(right);
                Vec3 dir = (right * u + up * v + forward).normalize();
                Vec3 col = trace(Ray(camPos, dir), tris, lightPos);
                SDL_SetRenderDrawColor(renderer, (Uint8)(col.x * 255), (Uint8)(col.y * 255), (Uint8)(col.z * 255), 255);
                SDL_RenderDrawPoint(renderer, x, y);
            }
        }

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
