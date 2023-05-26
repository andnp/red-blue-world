# This shouldn't need to be a class. No need to instantiate
# This should handle _when_ to load and _what_ to load
# This should also handle unloading
# Should be punted to a separate thread
#   - Thread, not process. Let the GIL be our locking mechanism
#   - Should be io bound and not compute bound
