import functools
import sys
__all__ = ['debug', 'export']

def export(fn):
	mod = sys.modules[fn.__module__]
	if hasattr(mod, '__all__'):
		if fn.__name__ not in mod.__all__:
			mod.__all__.append(fn.__name__)
	else:
		mod.__all__ = [fn.__name__]
	return fn

def debug(log=None):
	@functools.wraps(log)
	def decorator(func):
		"""Print the function signature and return value"""
		@functools.wraps(func)
		def wrapper_debug(*args, **kwargs):
			args_repr = [repr(a) for a in args]
			kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
			signature = ", ".join(args_repr + kwargs_repr)
			
			if log is not None:
				log.info(f"Calling {func.__name__}({signature})")
			else:
				print(f"Calling {func.__name__}({signature})")

			value = func(*args, **kwargs)
			
			if log is not None:
				log.info(f"{func.__name__!r} returned {value!r}")
				log.info("\n\n ---------------------------------------------- \n\n")
			else: 
				print(f"{func.__name__!r} returned {value!r}")
				print("\n\n ---------------------------------------------- \n\n")
		
			return value
		return wrapper_debug
	return decorator

# TODO implement decorator to split data in train, validation and test set given the dataset
def __minmax_maps(dataset):
	@functools.wraps(dataset)
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			train, valid, test = '', '', ''
			return train, valid, test
		return wrapper
	return decorator