import random

# Define a function that generates a random universal hash function
def generate_hash_function(self,shingle):
    # Define a prime number that is larger than the range of possible hash values
    p = 2147483647
    
    # Generate two random coefficients, a and b, in the range [0, p-1]
    a = random.randint(0, p-1)
    b = random.randint(0, p-1)
    
    # Define a hash function that takes an input key and returns a hash value
    def hash_function(key):
        return ((a * key + b) % p) % 200  # mod 200 to get a value in the range [0, 199]
    
    return hash_function

# Generate a list of 200 random hash functions
hash_functions = [generate_hash_function() for i in range(len(shingle))]
