from flask import Flask, render_template, request, jsonify
import sympy as sp
from sympy import symbols, sin, cos, exp, latex, simplify, expand, Function, solve, Eq, Symbol, S
from sympy.core.function import UndefinedFunction

app = Flask(__name__)

# Define the symbols
t = Symbol('t')
s = Symbol('s')
x = Function('x')
x_t = x(t)
dx_t = x_t.diff(t)
d2x_t = x_t.diff(t, 2)
def convert_to_hyperbolic(expr):
    """
    Convert exponential expressions to hyperbolic functions where possible.
    
    This function identifies and converts:
    - (e^x + e^-x)/2 -> cosh(x)
    - (e^x - e^-x)/2 -> sinh(x)
    - e^x -> cosh(x) + sinh(x)
    - e^-x -> cosh(x) - sinh(x)
    
    Args:
        expr: SymPy expression to convert
        
    Returns:
        SymPy expression with hyperbolic functions where possible
    """
    if isinstance(expr, (str, type(None))):
        return expr

    try:
        # First simplify the expression
        expr = sp.simplify(expr)
        
        # If already contains hyperbolic functions, return as is
        if any(func in str(expr) for func in ['sinh', 'cosh']):
            return expr
            
        def find_exponential_patterns(expression):
            """Find patterns that can be converted to hyperbolic functions."""
            patterns = []
            
            # Convert expression to expanded form for better pattern matching
            expanded = sp.expand(expression)
            
            if isinstance(expanded, sp.Add):
                terms = expanded.args
            else:
                terms = [expanded]
                
            # Look through all pairs of terms
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    try:
                        # Check if terms are exponentials
                        if isinstance(term1, sp.exp) and isinstance(term2, sp.exp):
                            arg1 = term1.args[0]
                            arg2 = term2.args[0]
                            
                            # Check if arguments are negatives of each other
                            if arg1 == -arg2:
                                coeff1 = sp.Wild('a')
                                coeff2 = sp.Wild('b')
                                pat1 = coeff1 * term1
                                pat2 = coeff2 * term2
                                
                                m1 = expression.match(pat1 + pat2)
                                if m1:
                                    if m1[coeff1] == m1[coeff2]:
                                        # Pattern for cosh
                                        patterns.append(('cosh', arg1, m1[coeff1], term1, term2))
                                    elif m1[coeff1] == -m1[coeff2]:
                                        # Pattern for sinh
                                        patterns.append(('sinh', arg1, m1[coeff1], term1, term2))
                    except:
                        continue
            
            return patterns
            
        # Function to recursively convert expressions
        def convert_recursive(expr):
            if isinstance(expr, sp.Add):
                return sp.Add(*[convert_recursive(arg) for arg in expr.args])
            elif isinstance(expr, sp.Mul):
                return sp.Mul(*[convert_recursive(arg) for arg in expr.args])
            elif isinstance(expr, sp.exp):
                # Convert single exponential using the formula e^x = cosh(x) + sinh(x)
                arg = expr.args[0]
                if arg.is_real:
                    return sp.cosh(arg) + sp.sinh(arg)
            return expr
            
        # First try to find and convert obvious patterns
        patterns = find_exponential_patterns(expr)
        
        # Apply the conversions
        for pat_type, arg, coeff, term1, term2 in patterns:
            if pat_type == 'cosh':
                expr = expr.subs(coeff * (term1 + term2), 2 * coeff * sp.cosh(arg))
            elif pat_type == 'sinh':
                expr = expr.subs(coeff * (term1 - term2), 2 * coeff * sp.sinh(arg))
                
        # Then try to convert remaining exponentials
        expr = convert_recursive(expr)
        
        # Final simplification
        return sp.simplify(expr)
        
    except Exception as e:
        # If any error occurs, return the original expression
        return expr

def get_explanation_for_ode(expr, solution):
    """
    Generate a detailed explanation based on the ODE characteristics.
    
    Args:
        expr: The ODE expression
        solution: The solution (can be None)
        
    Returns:
        str: A detailed explanation of the ODE
    """
    expr_str = str(expr)
    solution_str = str(solution) if solution else ""
    
    # Initialize explanation components
    order = 0
    is_homogeneous = True
    coefficients_type = "constant"  # can be "constant", "variable", or "nonlinear"
    forcing_terms = []
    
    try:
        # Determine order of the ODE
        if 'Derivative(x(t), t, 2)' in expr_str or 'd2x' in expr_str:
            order = 2
        elif 'Derivative(x(t), t)' in expr_str or 'dx' in expr_str:
            order = 1
            
        # Check if equation is homogeneous
        terms = expr_str.split(' + ')
        for term in terms:
            # Look for terms not involving x(t) or its derivatives
            if 't**' in term or 'sin(t)' in term or 'cos(t)' in term or 'exp' in term:
                if not any(x in term for x in ['x(t)', 'dx', 'd2x']):
                    is_homogeneous = False
                    forcing_terms.append(term)
        
        # Build the explanation
        explanation = []
        
        # Basic classification
        explanation.append(f"This is a {order}{'st' if order == 1 else 'nd'} order")
        explanation.append("linear" if 'x(t)**' not in expr_str else "nonlinear")
        explanation.append(f"{'homogeneous' if is_homogeneous else 'non-homogeneous'}")
        explanation.append("differential equation.")
        
        # Explain the structure
        if order == 1:
            explanation.append("\nIt's in the form dx/dt + P(t)x = Q(t),")
            explanation.append("where P(t) is the coefficient of x")
            explanation.append("and Q(t) is the forcing term." if not is_homogeneous else ".")
        elif order == 2:
            explanation.append("\nIt's in the form d²x/dt² + P(t)dx/dt + Q(t)x = R(t),")
            explanation.append("where P(t) and Q(t) are the coefficients")
            explanation.append("and R(t) is the forcing term." if not is_homogeneous else ".")
        
        # Explain forcing terms if present
        if not is_homogeneous:
            explanation.append("\nThe forcing term contains:")
            if any('t**' in term for term in forcing_terms):
                explanation.append("- Polynomial terms")
            if any('exp' in term for term in forcing_terms):
                explanation.append("- Exponential terms")
            if any('sin' in term or 'cos' in term for term in forcing_terms):
                explanation.append("- Trigonometric terms")
                
        # Explain solution characteristics if solution is provided
        if solution_str:
            explanation.append("\nThe solution contains:")
            if 'exp' in solution_str and not ('sinh' in solution_str or 'cosh' in solution_str):
                explanation.append("- Exponential terms from the characteristic equation")
            if 'sinh' in solution_str or 'cosh' in solution_str:
                explanation.append("- Hyperbolic functions (from real, distinct roots)")
            if ('sin' in solution_str or 'cos' in solution_str) and 'sinh' not in solution_str:
                explanation.append("- Trigonometric functions (from complex roots or forcing terms)")
            if 't**' in solution_str:
                explanation.append("- Polynomial terms (from the particular solution)")
                
        # Explain solution method
        explanation.append("\nThe Laplace transform method is particularly useful here because:")
        if not is_homogeneous:
            explanation.append("- It handles the forcing terms directly without variation of parameters")
        if 'exp' in expr_str:
            explanation.append("- It simplifies the handling of exponential terms")
        if order == 2:
            explanation.append("- It reduces the order of the differential equation")
        explanation.append("- It automatically incorporates the initial conditions")
        
        return " ".join(explanation)
        
    except Exception as e:
        # Fallback explanation if analysis fails
        return "This is a linear differential equation that can be solved using the Laplace transform method. " \
               "The method is particularly useful as it converts the differential equation into an algebraic equation, " \
               "making it easier to solve while automatically incorporating the initial conditions."

def parse_ode(expr_str):
    """Parse ODE expression with better handling of complex terms and equals signs."""
    try:
        # Split the equation if it contains equals sign
        if '=' in expr_str:
            left_side, right_side = expr_str.split('=')
            # Move everything to left side (subtract right side)
            expr_str = f"({left_side})-({right_side})"
        
        # Replace differential notation
        expr_str = expr_str.replace('d2x', 'Derivative(x(t), t, 2)')
        expr_str = expr_str.replace('dx', 'Derivative(x(t), t)')
        expr_str = expr_str.replace(' x ', ' x(t) ')
        
        # Handle case where x is at the start or end of string
        if expr_str.startswith('x '):
            expr_str = 'x(t) ' + expr_str[2:]
        if expr_str.endswith(' x'):
            expr_str = expr_str[:-2] + ' x(t)'
        if expr_str == 'x':
            expr_str = 'x(t)'
            
        # Parse the expression using sympy
        expr = sp.parse_expr(expr_str)
        return expr
    except Exception as e:
        raise ValueError(f"Error parsing expression: {str(e)}")

def convert_rational_coefficients(expr):
    """Convert rational numbers to sympy Rational for better handling."""
    if isinstance(expr, (int, float)):
        return S(expr)
    elif isinstance(expr, sp.Basic):
        return expr.replace(lambda x: isinstance(x, (int, float)), 
                          lambda x: S(x))
    return expr

def take_laplace_transform(expr, variable=t, transform_variable=s):
    """Takes the Laplace transform of an expression, handling special cases."""
    try:
        # Convert coefficients to Rational numbers
        expr = convert_rational_coefficients(expr)
        
        # Handle different types of expressions
        if isinstance(expr, sp.Add):
            # For sums, transform each term separately
            return sum(take_laplace_transform(term, variable, transform_variable) 
                     for term in expr.args)
        elif isinstance(expr, sp.Mul):
            # Handle special cases of products
            if any(isinstance(term, sp.exp) for term in expr.args):
                # Use direct Laplace transform for exponential terms
                result = sp.laplace_transform(expr, variable, transform_variable)
                return convert_rational_coefficients(result[0])
                
        # For other cases, use sympy's laplace_transform
        result = sp.laplace_transform(expr, variable, transform_variable)
        return convert_rational_coefficients(result[0])
    except Exception as e:
        raise ValueError(f"Error in Laplace transform: {str(e)}")

def solve_ode_with_steps(ode_expr, initial_conditions=None):
    """Solves an ODE using Laplace transforms with detailed steps."""
    if initial_conditions is None:
        initial_conditions = {}
        
    steps = []
    
    try:
        # Parse the ODE if it's a string
        if isinstance(ode_expr, str):
            ode_expr = parse_ode(ode_expr)
            
        # Convert coefficients to Rational numbers
        ode_expr = convert_rational_coefficients(ode_expr)
        
        # Rearrange equation to standard form
        if isinstance(ode_expr, sp.Equality):
            ode_expr = ode_expr.lhs - ode_expr.rhs
            
        # Add explanation of the ODE type
        explanation = get_explanation_for_ode(ode_expr, None)
        steps.append({
            'title': "Understanding the ODE",
            'explanation': explanation,
            'input': latex(ode_expr) + " = 0"
        })

        # Collect terms with x and its derivatives
        x_terms = sp.collect(ode_expr, [x_t, dx_t, d2x_t], evaluate=False)
        
        # Get forcing term (terms without x)
        forcing_term = -sum(term for var, term in x_terms.items() 
                          if not any(der in str(var) for der in ['x(t)', 'Derivative']))
        
        # Step 1: Take Laplace transform of each term
        steps.append({
            'title': "Taking the Laplace transform",
            'explanation': "Taking the Laplace transform of each term:",
            'input': latex(ode_expr) + " = 0"
        })

        # Take Laplace transform of the forcing term
        X = sp.Symbol('X')
        x0 = convert_rational_coefficients(initial_conditions.get(x_t, 0))
        dx0 = convert_rational_coefficients(initial_conditions.get(dx_t, 0))

        # Transform each term
        laplace_terms = {}
        for var, coeff in x_terms.items():
            coeff = convert_rational_coefficients(coeff)
            if str(var) == 'x(t)':
                laplace_terms[var] = coeff * X
            elif 'Derivative(x(t), t)' in str(var):
                laplace_terms[var] = coeff * (s*X - x0)
            elif 'Derivative(x(t), t, 2)' in str(var):
                laplace_terms[var] = coeff * (s**2*X - s*x0 - dx0)

        # Take Laplace transform of forcing term
        if forcing_term != 0:
            forcing_transform = take_laplace_transform(forcing_term)
        else:
            forcing_transform = 0

        # Combine all transformed terms
        laplace_expr = sp.expand(sum(laplace_terms.values()) - forcing_transform)
        
        steps.append({
            'title': "Substituting Laplace transforms",
            'explanation': f"Using:\nℒ{{x''(t)}} = s²X(s) - sx(0) - x'(0)\nℒ{{x'(t)}} = sX(s) - x(0)\nℒ{{x(t)}} = X(s)\nWith initial conditions: x(0) = {x0}, x'(0) = {dx0}",
            'result': latex(laplace_expr) + " = 0"
        })

        # Step 2: Solve for X(s)
        try:
            # Convert equation to polynomial form in X
            laplace_expr = sp.expand(laplace_expr)
            X_s = sp.solve(laplace_expr, X)[0]
            X_s = sp.simplify(X_s)
            
            steps.append({
                'title': "Solving for X(s)",
                'explanation': "Rearranging to isolate X(s):",
                'result': latex(sp.Eq(X, X_s))
            })
            
            # Step 3: Take inverse Laplace transform
            solution = sp.inverse_laplace_transform(X_s, s, t)
            solution = convert_to_hyperbolic(solution)
            solution = sp.simplify(solution)
            
            # Add final explanation
            final_explanation = get_explanation_for_ode(ode_expr, solution)
            
            steps.append({
                'title': "Taking the inverse Laplace transform",
                'explanation': final_explanation + "\nThe final solution x(t) is:",
                'result': latex(solution)
            })
            
            return {
                'steps': steps,
                'solution': latex(solution)
            }
        except Exception as e:
            raise ValueError(f"Error solving equation: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Error in solution process: {str(e)}")

# [Previous convert_to_hyperbolic and get_explanation_for_ode functions remain the same]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/laplace')
def laplace():
    return render_template('laplace.html')

@app.route('/inverse-laplace')
def inverse_laplace():
    return render_template('inverse_laplace.html')

@app.route('/ode')
def ode():
    return render_template('ode.html')

@app.route('/calculate_laplace', methods=['POST'])
def calculate_laplace():
    try:
        expr = request.json['expression']
        parsed_expr = sp.sympify(expr)
        result = sp.laplace_transform(parsed_expr, t, s)
        return jsonify({
            'success': True,
            'input': latex(parsed_expr),
            'result': latex(result[0])
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/calculate_inverse_laplace', methods=['POST'])
def calculate_inverse_laplace():
    try:
        expr = request.json['expression']
        parsed_expr = sp.sympify(expr)
        result = sp.inverse_laplace_transform(parsed_expr, s, t)
        result = convert_to_hyperbolic(result)
        return jsonify({
            'success': True,
            'input': latex(parsed_expr),
            'result': latex(result)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/solve_ode', methods=['POST'])
def solve_ode():
    try:
        data = request.json
        
        # Parse initial conditions
        initial_conditions = {}
        if 'y0' in data and data['y0']:  # Keep the form field names as y0/dy0 for compatibility
            try:
                initial_conditions[x_t] = float(data['y0'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Initial condition x(0) must be a number'
                })
                
        if 'dy0' in data and data['dy0']:
            try:
                initial_conditions[dx_t] = float(data['dy0'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Initial condition x\'(0) must be a number'
                })
            
        result = solve_ode_with_steps(data['equation'], initial_conditions)
        return jsonify({
            'success': True,
            'steps': result['steps'],
            'solution': result['solution']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)