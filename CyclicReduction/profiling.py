import sys
from typing import List

# def analyze_profiling_data(file_path: str, search_strings: List[str]) -> List[float]:
#     """
#     Calculates the operational cost percentage for a list of search strings
#     based on a profiling log file.

#     This function follows a simple logic:
#     1. The "total operations" is the sum of the numbers from every line in the file.
#     2. For each search string, its specific counter is the sum of the numbers from
#        every line where that string appears.
#     3. The final result is a list of percentages, calculated by dividing each
#        string's counter by the total operations.

#     Args:
#         file_path (str): The path to the input text file.
#         search_strings (List[str]): An ordered list of substrings to search for
#                                      in the function paths.

#     Returns:
#         List[float]: A list of percentages corresponding to the order of
#                      the input search_strings. Returns an empty list if the
#                      file cannot be read.
#     """
#     try:
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#     except (FileNotFoundError, Exception):
#         # In case of any error reading the file, return an empty list.
#         return []

#     total_operations = 0
#     # Initialize a dictionary to hold operation counts for each search string
#     search_ops_counter = {s: 0 for s in search_strings}

#     # Iterate through every line in the file
#     for line in lines:
#         clean_line = line.strip()
#         if not clean_line:
#             continue
            
#         parts = clean_line.rsplit(' ', 1)
#         if len(parts) != 2:
#             continue # Skip malformed lines

#         path, ops_str = parts
        
#         try:
#             ops_count = int(ops_str)
            
#             # 1. Add to the grand total operations from every line
#             total_operations += ops_count
            
#             # 2. Check if the line's path contains any of the search strings
#             for s_string in search_strings:
#                 if s_string in path:
#                     # If it does, add this line's ops to that string's counter
#                     search_ops_counter[s_string] += ops_count

#         except ValueError:
#             # Skip lines where the operation count is not a valid integer
#             continue

#     if total_operations == 0:
#         # Avoid division by zero; if no ops, all percentages are 0.
#         return [0.0] * len(search_strings)
#     print(total_operations)

#     # 3. Calculate percentages for each search string
#     percentages = []
#     for s_string in search_strings:
#         string_total_ops = search_ops_counter.get(s_string, 0)
#         print(string_total_ops)
#         percentage = (string_total_ops / total_operations) * 100
#         percentages.append(percentage)
        
#     return percentages

def analyze_profiling_data(file_path: str, search_strings: list[str]) -> dict[str, float]:
    """
    Analyzes a PETSc flamegraph log file to calculate the operational cost
    percentage for a list of specified function names.

    The logic is as follows:
    1. It only considers lines that start with 'Main '.
    2. The "total operations" is the sum of the numbers from every such line.
    3. For each search string, its counter is the sum of the numbers from lines
       where the string appears in the function path (the "middlestring").
    4. The final result is a dictionary where keys are the search strings and
       values are their operational cost percentages.

    Args:
        file_path (str): The path to the input PETSc log file.
        search_strings (List[str]): An ordered list of substrings to search for
                                    in the function paths.

    Returns:
        Dict[str, float]: A dictionary mapping each search string to its
                          calculated percentage. Returns an empty dictionary
                          if the file cannot be read or no operations are found.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        return {}

    total_operations = 0
    # Initialize a dictionary to hold operation counts for each search string
    search_ops_counter = {s: 0 for s in search_strings}
    prefix = 'Main '

    # Iterate through every line in the file
    for line in lines:
        clean_line = line.strip()
        
        # 1. We only care about lines that represent a profiled function call
        if not clean_line.startswith(prefix):
            continue
            
        # 2. Split the line into the path part and the number part.
        # rsplit is used because the path itself may contain spaces.
        parts = clean_line.rsplit(' ', 1)
        if len(parts) != 2:
            continue # Skip malformed lines

        path_part, ops_str = parts
        
        # 3. Ensure the operation count is a valid number
        try:
            ops_count = int(ops_str)
        except ValueError:
            continue

        # 4. Extract the "middlestring" (the actual function path) by removing the prefix
        middlestring = path_part[len(prefix):]
        
        # 5. Accumulate the counts
        # Add to the grand total operations from every valid line
        total_operations += ops_count
        
        # Check if the line's path contains any of the search strings
        for s_string in search_strings:
            if s_string in middlestring:
                # If it does, add this line's ops to that string's counter
                search_ops_counter[s_string] += ops_count

    if total_operations == 0:
        # Avoid division by zero; if no ops, all percentages are 0.
        print("Warning: Total operations found in the log file is zero.", file=sys.stderr)
        return {s: 0.0 for s in search_strings}

    # 6. Calculate percentages and store them in a dictionary
    percentages = []
    for s_string in search_strings:
        string_total_ops = search_ops_counter.get(s_string, 0)
        percentage = (string_total_ops / total_operations) * 100
        percentages.append(percentage)
        
    return percentages

# flamegraph_search_terms = ["startup", "forward_reduction", "interface_solve",
#                                             "forward_substitution", "solution_filling",]

# print(analyze_profiling_data("/home/shomea/t/tinuskg/WRMG/RelaxNavierStokesFork/metrics_folder/heat_run_log.txt", flamegraph_search_terms))

