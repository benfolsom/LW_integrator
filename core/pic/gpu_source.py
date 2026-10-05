"""Shared strict C bodies adapted from the measured GPU spike.

The atomic TwoSum correction and compensated gather are retained. Native PIC
uses open, rectangular grids, cell charge (C), and SI Boris kicks. No optional
framework is imported here. Host C emulation exercises these same bodies.
"""


def header(metal: bool = False, dtype: str = "float32") -> str:
    qualifier = "" if metal else "__device__ "
    pointer = "thread " if metal else ""
    return (
        (
            "typedef float T;\n"
            if metal
            else (
                "#ifndef __CUDACC_RTC__\n#include <math.h>\n#endif\n"
                f"typedef {'double' if dtype == 'float64' else 'float'} T;\n"
            )
        )
        + f"""
{qualifier}inline void shape_weights(T x, int order, {pointer}int& base,
                                       {pointer}T* w) {{
    if(order==1) {{
        base=int(floor(x)); T f=x-T(base); w[0]=T(1)-f; w[1]=f;
    }} else {{
        int mid=int(floor(x+T(0.5))); T d=x-T(mid); base=mid-1;
        w[0]=T(0.5)*(T(0.5)-d)*(T(0.5)-d);
        w[1]=T(0.75)-d*d; w[2]=T(0.5)*(T(0.5)+d)*(T(0.5)+d);
    }}
}}
"""
    )


def atomic_add(target: str, index: str, value: str, metal: bool = False) -> str:
    def add(name: str, val: str) -> str:
        if metal:
            return (
                f"atomic_fetch_add_explicit(&{name}[{index}], "
                f"{val}, memory_order_relaxed)"
            )
        return f"atomicAdd(&{name}[{index}], {val})"

    return f"""{{
        T v={value}; T old={add('high' + target, 'v')};
        T sum=old+v; T z=sum-old;
        T err=(old-(sum-z))+(v-z);
        {add('low' + target, 'err')};
    }}"""


STENCIL = """
int base[3]; T w[3][3];
for(int a=0;a<3;++a) shape_weights(x[3*p+a],meta[3],base[a],w[a]);
"""


def deposit_body(metal: bool = False) -> str:
    return STENCIL + """
for(int a=0;a<=meta[3];++a) for(int b=0;b<=meta[3];++b)
for(int c=0;c<=meta[3];++c) {
    int cell=((base[0]+a)*meta[1]+base[1]+b)*meta[2]+base[2]+c;
""" + atomic_add("", "cell", "q[p]*w[0][a]*w[1][b]*w[2][c]", metal) + "}"


GATHER = STENCIL + """
for(int d=0;d<meta[4];++d) {
    T sum=0, corr=0;
    for(int a=0;a<=meta[3];++a) for(int b=0;b<=meta[3];++b)
    for(int c=0;c<=meta[3];++c) {
        int cell=((base[0]+a)*meta[1]+base[1]+b)*meta[2]+base[2]+c;
        T y=field[meta[4]*cell+d]*w[0][a]*w[1][b]*w[2][c]-corr;
        T next=sum+y; corr=(next-sum)-y; sum=next;
    }
    result[meta[4]*p+d]=sum;
}
"""

PUSH = """
T half_step=qm[p]*params[0]/T(2), minus[3], t[3], prime[3], norm=0, tn=0;
for(int a=0;a<3;++a) {
    minus[a]=u[3*p+a]+half_step*electric[3*p+a]/T(299792458.0);
    norm+=minus[a]*minus[a];
}
T gamma=sqrt(T(1)+norm);
for(int a=0;a<3;++a) { t[a]=half_step*magnetic[3*p+a]/gamma; tn+=t[a]*t[a]; }
for(int a=0;a<3;++a) {
    int b=(a+1)%3,c=(a+2)%3;
    prime[a]=minus[a]+(minus[b]*t[c]-minus[c]*t[b]);
}
for(int a=0;a<3;++a) {
    int b=(a+1)%3,c=(a+2)%3;
    T sb=T(2)*t[b]/(T(1)+tn), sc=T(2)*t[c]/(T(1)+tn);
    result[3*p+a]=minus[a]+(prime[b]*sc-prime[c]*sb)
                  +half_step*electric[3*p+a]/T(299792458.0);
}
"""

# One thread per observer: coefficient tables are generated in host float64
# to avoid subtractive cancellation in the rectangular-cell Green primitive.
NODES = """
for(int a=0;a<3;++a) {
    T sum=0,corr=0;
    for(int j=0;j<meta[0];++j) {
        T y=table[(p*meta[0]+j)*3+a]*q[p*meta[0]+j]-corr;
        T next=sum+y; corr=(next-sum)-y; sum=next;
    }
    result[3*p+a]=sum;
}
"""


def current_body(metal: bool = False) -> str:
    body = """
T delta[3], extent=0;
for(int a=0;a<3;++a) { delta[a]=newx[3*p+a]-oldx[3*p+a];
    T v=delta[a]<0 ? -delta[a] : delta[a]; if(v>extent) extent=v; }
int segments=int(ceil(extent)); if(segments<1) segments=1;
for(int seg=0;seg<segments;++seg) {
    int start[3],length[3]; T s[3][5],ds[3][5];
    for(int a=0;a<3;++a) {
        T lo=oldx[3*p+a]+delta[a]*(T(seg)/T(segments));
        T hi=oldx[3*p+a]+delta[a]*(T(seg+1)/T(segments));
        int b0,b1; T w0[3],w1[3];
        shape_weights(lo,meta[3],b0,w0); shape_weights(hi,meta[3],b1,w1);
        start[a]=b0<b1 ? b0:b1;
        length[a]=(b0>b1 ? b0:b1)+meta[3]+1-start[a];
        for(int j=0;j<5;++j) { s[a][j]=0;ds[a][j]=0; }
        for(int j=0;j<=meta[3];++j) {
            s[a][b0-start[a]+j]=w0[j]; ds[a][b1-start[a]+j]=w1[j]; }
        for(int j=0;j<length[a];++j) ds[a][j]-=s[a][j];
    }
    T volume=params[0]*params[1]*params[2];
    for(int axis=0;axis<3;++axis) {
        int baxis=(axis+1)%3,caxis=(axis+2)%3; T cumulative=0;
        for(int a=0;a<length[axis];++a) {
            cumulative+=ds[axis][a];
            for(int b=0;b<length[baxis];++b) for(int c=0;c<length[caxis];++c) {
                T transverse=s[baxis][b]*s[caxis][c]
                    +T(0.5)*ds[baxis][b]*s[caxis][c]
                    +T(0.5)*s[baxis][b]*ds[caxis][c]
                    +ds[baxis][b]*ds[caxis][c]/T(3);
                T value=-q[p]*params[axis]/(volume*params[3])*(cumulative*transverse);
                int idx[3]={start[0],start[1],start[2]};
                idx[axis]+=a+1;idx[baxis]+=b;idx[caxis]+=c;
"""
    for axis in range(3):
        ny = "(meta[1]+1)" if axis == 1 else "meta[1]"
        nz = "(meta[2]+1)" if axis == 2 else "meta[2]"
        body += f"if(axis=={axis}) {{ int cell=(idx[0]*{ny}+idx[1])*{nz}+idx[2];"
        body += atomic_add(str(axis), "cell", "value", metal) + "}\n"
    return body + "}}}}"


def operations(
    metal: bool = False,
) -> dict[str, tuple[list[str], list[str], str, bool]]:
    return {
        "deposit": (["x", "q", "meta"], ["high", "low"], deposit_body(metal), True),
        "gather": (["x", "field", "meta"], ["result"], GATHER, False),
        "push": (
            ["u", "electric", "magnetic", "qm", "params"],
            ["result"],
            PUSH,
            False,
        ),
        "current": (
            ["oldx", "newx", "q", "meta", "params"],
            ["high0", "low0", "high1", "low1", "high2", "low2"],
            current_body(metal),
            True,
        ),
        "nodes": (["table", "q", "meta"], ["result"], NODES, False),
    }


def cuda_source(dtype: str = "float64") -> str:
    source = header(False, dtype)
    for name, (inputs, outputs, body, _) in operations().items():
        args = [f"const {'int' if arg == 'meta' else 'T'}* {arg}" for arg in inputs]
        args += [f"T* {arg}" for arg in outputs] + ["int count"]
        source += f'extern "C" __global__ void {name}({", ".join(args)}) {{\n'
        source += "int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count) return;\n"
        source += body + "\n}\n"
    return source
