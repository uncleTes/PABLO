from functools import wraps

def dump_args(func):
	"""This decorator dumps out the *args passed to a function
	before calling it"""

	@wraps(func)
	def echo_func(*args,**kwargs):
		print(func.__name__ + "(" + ", ".join(str(arg) for arg in args)
			+ ")")
		return func(*args, **kwargs)

	return echo_func 
