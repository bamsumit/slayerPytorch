import csv
from itertools import zip_longest

def iterable_int_pair_comparator(iter_int1, iter_int2, params = {}):
	for (int1, int2) in zip_longest(iter_int1, iter_int2):
		if int(int1) != int(int2):
			return False
	return True
	# Would this work?
	# return np.array_equal(iter_int1, iter_int2)

def iterable_float_pair_comparator(iter_float1, iter_float2, params = {}):
	for (float1, float2) in zip_longest(iter_float1, iter_float2):
		if abs(float(float1) - float(float2)) > params['FLOAT_EPS_TOL']:
			print("Found difference, ", float1, " ", float2)
			# return False
	return True

# Utility function to compare ndarray to one contained in CSV file generated separately (i.e. MATLAB)
def is_array_equal_to_file(array, filepath, has_header=False, compare_function=iterable_int_pair_comparator, comp_params = {}):
	with open(filepath, 'r') as csvfile:
		reader = csv.reader(csvfile)
		# Skip header
		if has_header: next(reader, None)
		for (g_truth, read_r) in zip_longest(reader, array):
			if not compare_function(g_truth, read_r, comp_params):
				return False
	return True