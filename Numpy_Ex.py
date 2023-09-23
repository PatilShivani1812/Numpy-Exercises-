# # #1. Import the numpy package under the name `np` 
import numpy as np

# # #2. Print the numpy version and the configuration 
# print(np.__version__)
# np.show_config()

# # # 3. Create a null vector of size 10
Z = np.zeros(10)
print(Z)

# # #4. How to find the memory size of any array
Z = np.zeros((10,10))
print("%d bytes" %(Z.size * Z.itemsize))

# #5. How to get the documentation of the numpy add function from the command line?
# #python -c "import numpy; help(numpy.add)"

# # 6. Create a null vector of size 10 but the fifth value which is 1 
Z = np.zeros(10)
Z[4] = 1
print(Z)

# #7. Create a vector with values ranging from 10 to 49
z = np.arange(10,50)
print(z)

# #8. Reverse a vector (first element becomes last)
z = np.arange(50)
y = z[::-1]
print(y)

# # 9. Create a 3x3 matrix with values ranging from 0 to 8
z = np.arange(9).reshape(3,3)
print(z)

# # 10. Find indices of non-zero elements from [1,2,0,0,4,0]
nz = np.nonzero([1,2,0,0,4,0])
print(nz)

# # 11. Create a 3x3 identity matrix
z = np.eye(3)
print(z)

# #12. Create a 3x3x3 array with random values
z = np.random.random((3,3,3))
print(z)

# #13. Create a 10x10 array with random values and find the minimum and maximum values
z = np.random.random((10,10))
zmin = z.min()
zmax =z.max()
print(zmin)
print(zmax)

# # 14. Create a random vector of size 30 and find the mean value
z = np.random.random(30)
m = z.mean()
print(m)

# #15. Create a 2d array with 1 on the border and 0 inside
z = np.ones((10,10))
z[1:-1,1:-1] = 0
print(z)

# #16. How to add a border (filled with 0's) around an existing array?
z = np.ones((5,5))
z = np.pad(z,pad_width = 1,mode = "constant" , constant_values = 0)
print(z)

# # Using fancy indexing
z[:,[0,-1]] = 0
z[[0,-1],:] = 0
print(z)

# # 17. What is the result of the following expression? 
print(0 * np.nan)  #nan = not a number
print(np.nan == np.nan) #False
print(np.inf > np.nan) #False
print(np.nan in set([np.nan]))#True
print(0.3 == 3 * 0.1)#False
# #Note =NaN = not a number, inf = infinity

# #18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal 
z = np.diag(1+np.arange(4),k = -1)
print(z)

# #19. Create a 8x8 matrix and fill it with a checkerboard pattern
z = np.zeros((8,8),dtype = int)
z[1::2,::2] = 1
z[::2,1::2] = 1
print(z)

# #20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
print(np.unravel_index(99,(6,7,8)))

# #21. Create a checkerboard 8x8 matrix using the tile function
z = np.tile(np.array([[0,1],[1,0]]),(4,4))
print(z)

# #22. Normalize a 5x5 random matrix
z = np.random.random((5,5))
z = (z-np.mean(z)) / (np.std(z))
print(z)

# # 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) 

color = np.dtype([
    ("r", np.ubyte),
    ("g", np.ubyte),
    ("b", np.ubyte),
    ("a", np.ubyte)
])

#24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) 
z = np.dot(np.ones((5,3)),np.ones((3,2)))
print(z)

# # Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)

# #25. Given a 1D array, negate all elements which are between 3 and 8, in place.
z = np.arange(11)
z[(3<z)&(z<8)]*= -1
print(z)

# #26. What is the output of the following script?
print(sum(range(5),-1))
from numpy import*
print(sum(range(5),-1))

# #27. What are the result of the following expressions?
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

# #28 How to get the dates of yesterday, today and tomorrow? 
yesterday = np.datetime64('today') - np.timedelta64(1)
print(yesterday)
today = np.datetime64('today')
print(today)
tomorrow  = np.datetime64('today') + np.timedelta64(1)
print(tomorrow)

# #29. How to get all the dates corresponding to the month of July 2016? 
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)

# #30. Create a 5x5 matrix with row values ranging from 0 to 4 
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

# #31. Consider a generator function that generates 10 integers and use it to build an array
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)

# #32. Create a vector of size 10 with values ranging from 0 to 1, both excluded 
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)

# #33. Create a random vector of size 10 and sort it (★★☆)
Z = np.random.random(10)
Z.sort()
print(Z)

# #34. How to sum a small array faster than np.sum? 
Z = np.arange(10)
z= np.add.reduce(Z)
print(z)

# #35. Consider two random array A and B, check if they are equal
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# # Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# # Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)

# #36. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

# #37. How to convert a float (32 bits) array into an integer (32 bits) in place?
z = (np.random.rand(10)*100).astype(np.float32)
y = z.view(np.int32)
y[:] = z
print(y)

# #38. How to sort an array by the nth column?
z = np.random.randint(0,10,(3,3))
print(z)
print(z[z[:,1].argsort()])
