import numpy as np
from nose.tools import assert_true
from scipy.sparse import csr_matrix

def test_0():
    print ('in should Convert a float array correctly')
    assert_true(np.allclose(np.load("/tmp/test.npy"),[0.3,0.5]))
 
def test_1():
    print ('in should Convert a sequence of integers correctly')
    assert_true(np.allclose(np.load("/tmp/inttest.npy"),[1,2,3,4,5,6,7,8,9,10]))

def test_3():
	print ('in should write bytes directly')
	assert_true(np.allclose(np.load("/tmp/inttest3.npy"),[1,2,3,4,5,6,7,8,9,10]))

def test_5():
	print ('in should work with csr matrix')
	loader=np.load("/tmp/data.npz")
	mat=csr_matrix((loader['data'],loader['indices'],loader['indptr'])).toarray()
	assert_true(np.allclose(mat,np.asarray([[ 0.1 ,  0,0       ],[ 0,  0.2,0],[ 0 ,0,0.3       ]], dtype=np.float32)))