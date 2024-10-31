import sympy as sp
from sympy import symbols, sin, cos, exp, latex, simplify, expand, Function, solve, Eq
from sympy.core.function import UndefinedFunction

# Define the symbols
t, s = sp.symbols('t s')
y = Function('y')(t)
dy = y.diff(t)
d2y = y.diff(t, 2)

def get_explanation_for_ode(expr, solution):
    """Generate a personalized explanation based on the type of ODE."""
    expr_str = str(expr)
    solution_str = str(solution)
    
    explanation = ""
    
    # Check for different types of ODEs and provide specific explanations
    if 'sin(t)' in expr_str or 'cos(t)' in expr_str:
        explanation += "This is a non-homogeneous ODE with trigonometric forcing terms. "
        explanation += "The solution combines the general solution of the homogeneous equation "
        explanation += "with a particular solution matching the frequency of the input."
    
    if 't' in expr_str and 't**' not in expr_str and 'sin' not in expr_str and 'cos' not in expr_str:
        explanation += "This is a non-homogeneous ODE with a linear forcing term. "
        explanation += "The particular solution includes a linear term to match the input."
    
    if 't**' in expr_str:
        degree = max([int(term.split('**')[1]) for term in expr_str.split() if 't**' in term])
        explanation += f"This is a non-homogeneous ODE with a polynomial forcing term of degree {degree}. "
        explanation += f"The particular solution includes terms up to t^{degree}."
    
    if 'sinh' in solution_str or 'cosh' in solution_str:
        explanation += "The solution involves hyperbolic functions, which is typical for ODEs "
        explanation += "with characteristic equations having real, distinct roots."
    
    if 'sin' in solution_str or 'cos' in solution_str:
        if 'sin' not in expr_str and 'cos' not in expr_str:
            explanation += "The solution involves trigonometric functions, which is typical for ODEs "
            explanation += "with characteristic equations having complex roots."
    
    if not explanation:
        explanation = "This is a linear, constant-coefficient differential equation. "
        explanation += "The solution form depends on the roots of the characteristic equation."
    
    return explanation

def convert_to_hyperbolic(expr):
    """Convert exponential expressions to hyperbolic functions where possible."""
    # Define basic patterns for hyperbolic functions
    def find_sinh():
        matches = []
        # Look for patterns like (e^x - e^-x)/2
        terms = expr.expand().args if hasattr(expr.expand(), 'args') else [expr]
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                try:
                    # Check for exp(x) and -exp(-x) patterns
                    if (isinstance(term1, sp.exp) and isinstance(term2, sp.exp)):
                        arg1 = term1.args[0]
                        arg2 = term2.args[0]
                        if arg1 == -arg2:
                            matches.append((term1, term2, arg1))
                except:
                    continue
        return matches

    def find_cosh():
        matches = []
        # Look for patterns like (e^x + e^-x)/2
        terms = expr.expand().args if hasattr(expr.expand(), 'args') else [expr]
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                try:
                    # Check for exp(x) and exp(-x) patterns
                    if (isinstance(term1, sp.exp) and isinstance(term2, sp.exp)):
                        arg1 = term1.args[0]
                        arg2 = term2.args[0]
                        if arg1 == -arg2:
                            matches.append((term1, term2, arg1))
                except:
                    continue
        return matches

    # Try to simplify the expression first
    expr = simplify(expr)
    
    # If the expression already contains sinh or cosh, return as is
    if 'sinh' in str(expr) or 'cosh' in str(expr):
        return expr

    # Try to identify and replace patterns
    expanded = expand(expr)
    
    # Look for sinh patterns
    sinh_matches = find_sinh()
    for term1, term2, arg in sinh_matches:
        try:
            expr = expr.subs(term1 - term2, 2 * sp.sinh(arg))
        except:
            continue

    # Look for cosh patterns
    cosh_matches = find_cosh()
    for term1, term2, arg in cosh_matches:
        try:
            expr = expr.subs(term1 + term2, 2 * sp.cosh(arg))
        except:
            continue

    return simplify(expr)

def solve_ode_with_steps(ode_expr, initial_conditions=None):
    """Solves an ODE using Laplace transforms with detailed steps."""
    steps = []
    
    # Add explanation of the ODE type
    explanation = get_explanation_for_ode(ode_expr, None)
    steps.append({
        'title': "Understanding the ODE",
        'explanation': explanation,
        'input': latex(ode_expr)
    })
    
    # Step 1: Parse the ODE and take Laplace transform
    steps.append({
        'title': "Taking the Laplace transform of both sides",
        'explanation': "Using the linearity property and transform rules for derivatives:",
        'input': latex(ode_expr)
    })
    
    # Replace derivatives with Laplace transform formulas
    Y = sp.Symbol('Y')
    laplace_expr = ode_expr.subs({
        d2y: s**2 * Y - s * initial_conditions.get(y, 0) - initial_conditions.get(dy, 0),
        dy: s * Y - initial_conditions.get(y, 0),
        y: Y
    })
    
    steps.append({
        'title': "Substituting Laplace transforms",
        'explanation': "Using:\nℒ{y''(t)} = s²Y(s) - sy(0) - y'(0)\nℒ{y'(t)} = sY(s) - y(0)\nℒ{y(t)} = Y(s)",
        'result': latex(laplace_expr)
    })
    
    # Step 2: Solve for Y(s)
    Y_s = solve(laplace_expr, Y)[0]
    Y_s = simplify(Y_s)
    
    steps.append({
        'title': "Solving for Y(s)",
        'explanation': "Rearranging to isolate Y(s):",
        'result': latex(sp.Eq(Y, Y_s))
    })
    
    # Step 3: Take inverse Laplace transform
    solution = sp.inverse_laplace_transform(Y_s, s, t)
    solution = convert_to_hyperbolic(solution)
    
    # Add final explanation based on the solution
    final_explanation = get_explanation_for_ode(ode_expr, solution)
    
    steps.append({
        'title': "Taking the inverse Laplace transform",
        'explanation': final_explanation + "\nThe final solution y(t) is:",
        'result': latex(solution)
    })
    
    return {
        'steps': steps,
        'solution': latex(solution)
    }