"""Small explicit C/Metal kernel bodies; arrays have C-order, xyz-last layout.

Sharing body text avoids gratuitous duplication between CuPy and MLX. Taichi
is a separate implementation. No framework is imported by this module.
"""


def stencil_source(grid, order):
    return f"""
    const int n = {grid.n};
    const T dx = T({grid.dx!r});
    int ids[3][3]; T w[3][3];
    for (int a = 0; a < 3; ++a) {{
        T s = x[3*p+a] / dx;
        s -= floor(s / T(n)) * T(n);
        int base;
        if ({order} == 1) {{
            base = int(floor(s)); T f = s - T(base);
            w[a][0] = T(1) - f; w[a][1] = f;
        }} else {{
            int center = int(floor(s + T(0.5))); T d = s - T(center);
            base = center - 1;
            w[a][0] = T(0.5)*(T(0.5)-d)*(T(0.5)-d);
            w[a][1] = T(0.75)-d*d;
            w[a][2] = T(0.5)*(T(0.5)+d)*(T(0.5)+d);
        }}
        for (int j = 0; j <= {order}; ++j)
            ids[a][j] = (base+j+n)%n;
    }}
    """


def deposit_body(grid, order, metal=False, compensated=True):
    add = (
        "atomic_fetch_add_explicit(&high[cell], v, memory_order_relaxed)"
        if metal
        else "atomicAdd(&high[cell], v)"
    )
    correction = ""
    if compensated:
        # The atomic operation returns the value immediately before OUR add.
        # Reconstruct that add's roundoff with TwoSum and accumulate it in low.
        # Both additions remain nondeterministic; this is compensation, not f64.
        low_add = (
            "atomic_fetch_add_explicit(&low[cell], err, memory_order_relaxed);"
            if metal
            else "atomicAdd(&low[cell], err);"
        )
        correction = f"""
        T sum = old + v;
        T z = sum - old;
        T err = (old - (sum - z)) + (v - z);
        {low_add}
        """
    return stencil_source(grid, order) + f"""
    for (int i=0; i<={order}; ++i)
    for (int j=0; j<={order}; ++j)
    for (int k=0; k<={order}; ++k) {{
        int cell = (ids[0][i]*n + ids[1][j])*n + ids[2][k];
        T v = q[p]*w[0][i]*w[1][j]*w[2][k]/(dx*dx*dx);
        T old = {add};
        {correction}
    }}
    """


def gather_body(grid, order):
    return stencil_source(grid, order) + f"""
    T sum[3] = {{0,0,0}}, correction[3] = {{0,0,0}};
    for (int i=0; i<={order}; ++i)
    for (int j=0; j<={order}; ++j)
    for (int k=0; k<={order}; ++k) {{
        int cell = (ids[0][i]*n + ids[1][j])*n + ids[2][k];
        T weight = w[0][i]*w[1][j]*w[2][k];
        for (int a=0; a<3; ++a) {{
            T y = weight*field[3*cell+a]-correction[a];
            T updated = sum[a]+y;
            correction[a] = (updated-sum[a])-y;
            sum[a] = updated;
        }}
    }}
    for (int a=0; a<3; ++a) result[3*p+a]=sum[a];
    """


def push_body(grid):
    return f"""
    T kick = qm[p]*dt[0]*T(0.5);
    T minus[3], t[3], prime[3], out[3];
    T norm = 0;
    for (int a=0; a<3; ++a) {{
        minus[a] = u[3*p+a]+kick*electric[3*p+a];
        norm += minus[a]*minus[a];
    }}
    T gamma = sqrt(T(1)+norm), tnorm=0;
    for (int a=0; a<3; ++a) {{
        t[a] = kick*magnetic[3*p+a]/gamma;
        tnorm += t[a]*t[a];
    }}
    for (int a=0; a<3; ++a) {{
        int b=(a+1)%3, c=(a+2)%3;
        prime[a]=minus[a]+minus[b]*t[c]-minus[c]*t[b];
    }}
    norm=0;
    for (int a=0; a<3; ++a) {{
        int b=(a+1)%3, c=(a+2)%3;
        out[a]=minus[a]+T(2)/(T(1)+tnorm)*(prime[b]*t[c]-prime[c]*t[b])
               +kick*electric[3*p+a];
        norm += out[a]*out[a];
        un[3*p+a]=out[a];
    }}
    gamma=sqrt(T(1)+norm);
    for (int a=0; a<3; ++a) {{
        T pos=x[3*p+a]+dt[0]*out[a]/gamma;
        xn[3*p+a]=pos-floor(pos/T({grid.length!r}))*T({grid.length!r});
    }}
    """


def cuda_source(grid, dtype):
    ctype = "double" if dtype == "float64" else "float"
    header = f"#ifndef __CUDACC_RTC__\n#include <math.h>\n#endif\ntypedef {ctype} T;\n"
    kernels = []
    for order in (1, 2):
        kernels.append(f"""extern "C" __global__ void deposit{order}(
            const T* x, const T* q, T* high, T* low, int count) {{
            int p=blockIdx.x*blockDim.x+threadIdx.x;
            if(p>=count) return;
            {deposit_body(grid, order)}
        }}""")
        kernels.append(f"""extern "C" __global__ void gather{order}(
            const T* x, const T* field, T* result, int count) {{
            int p=blockIdx.x*blockDim.x+threadIdx.x;
            if(p>=count) return;
            {gather_body(grid, order)}
        }}""")
    kernels.append(f"""extern "C" __global__ void push(
        const T* x, const T* u, const T* electric, const T* magnetic,
        const T* qm, const T* dt, T* xn, T* un, int count) {{
        int p=blockIdx.x*blockDim.x+threadIdx.x;
        if(p>=count) return;
        {push_body(grid)}
    }}""")
    return header + "\n".join(kernels)
