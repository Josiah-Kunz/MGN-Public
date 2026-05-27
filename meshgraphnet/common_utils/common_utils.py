import re   # regex

def format_name(name, strip_units=False):
    """
    Format variable names for display.
    
    Converts snake_case, camelCase, etc. to Title Case with spaces.
    Preserves units in parentheses without capitalizing them.
    
    Parameters:
    -----------
    name : str
        Variable name to format
    strip_units : bool, optional
        If True, remove units in parentheses (default: False)
    
    Examples:
    ---------
    'von_mises' -> 'Von Mises'
    'vonMises' -> 'Von Mises'
    'u_magnitude' -> 'U Magnitude'
    'sigma_xx' -> 'Sigma Xx'
    'x (m)' -> 'X (m)'
    'x (m)', strip_units=True -> 'X'
    'stress (MPa)' -> 'Stress (MPa)'
    'stress (MPa)', strip_units=True -> 'Stress'
    'von_mises (Pa)' -> 'Von Mises (Pa)'
    'von_mises (Pa)', strip_units=True -> 'Von Mises'
    """
    # Check if there are units in parentheses at the end
    unit_match = re.search(r'\s*\(([^)]+)\)$', name)

    if unit_match:
        # Split into base name and unit
        base_name = name[:unit_match.start()]
        unit = unit_match.group(1)

        # Format the base name
        formatted_base = _format_base_name(base_name)

        # Return with or without units based on strip_units flag
        if strip_units:
            return formatted_base
        else:
            return f"{formatted_base} ({unit})"
    else:
        # No units, just format normally
        return _format_base_name(name)


def _format_base_name(name):
    """
    Helper function to format the base name without units.
    
    Parameters:
    -----------
    name : str
        Base name without units
        
    Returns:
    --------
    str
        Formatted name in Title Case
    """
    # Split on underscores
    name = name.replace('_', ' ')

    # Split camelCase: insert space before capitals
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

    # Title case
    name = name.title()

    return name