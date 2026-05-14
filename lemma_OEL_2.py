import sympy as sp

# Define symbols
n, p = sp.symbols('n p')

# Function to calculate raw moments of a Binomial distribution B(N, p)
def binom_moment(N, k):
    t = sp.symbols('t')
    # Moment generating function for Binomial: M(t) = (1 - p + p*e^t)^N
    M = (1 - p + p*sp.exp(t))**N
    # k-th derivative w.r.t t, evaluated at t=0
    return sp.diff(M, t, k).subs(t, 0).simplify()

# Calculate the required moments from your formula
t1_1 = binom_moment(n-1, 2)            # E[ B(n-1, p)^2 ]
t1_2 = binom_moment((n-1)**2, 2)       # E[ B((n-1)^2, p)^2 ]
t2   = binom_moment(n-1, 4)            # E[ B(n-1, p)^4 ]
t3_1 = binom_moment(n-1, 3)            # E[ B(n-1, p)^3 ]
t3_2 = binom_moment((n-1)**2, 1)       # E[ B((n-1)^2, p) ]
t4   = binom_moment(n*(n-1), 4)        # E[ B(n(n-1), p)^4 ]

# Assemble your final equation
E_sigma2_M2 = (1/n**2) * (t1_1 * t1_2 + t2 + 2 * t3_1 * t3_2) - (1/n**4) * t4

# Simplify the results
result = sp.simplify(E_sigma2_M2)

print("Expanded Polynomial:")
print(sp.expand(result))
print("\nFactored Form:")
print(sp.factor(result))