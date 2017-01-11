import numpy as np
from nose.tools import assert_true
from scipy.sparse import csr_matrix

def test_0():
    print ('in should Convert a float array correctly')
    assert_true(np.allclose(np.load("/tmp/test.npy"),[0.3,0.5]))
 
def test_1():
    print ('in should Convert a sequence of integers correctly')
    assert_true(np.allclose(np.load("/tmp/inttest.npy"),[1,2,3,4,5,6,7,8,9,10]))

def test_2():
    print ('in should Convert a sequence to non 1- indexed itegers correctly')
    assert_true(np.allclose(np.load("/tmp/inttest2.npy"),[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))

def test_3():
	print ('in should write bytes directly')
	assert_true(np.allclose(np.load("/tmp/inttest3.npy"),[1,2,3,4,5,6,7,8,9,10]))

def test_4():
	print ('in should write bytes directly in a sequence')
	assert_true(np.allclose(np.load("/tmp/inttest4.npy"),[11,12,13,14,15,16,17,18,19,20]))

def test_5():
	print ('in should work with csr matrix')
	loader=np.load("/tmp/data.npz")
	mat=csr_matrix((loader['data'],loader['indices'],loader['indptr'])).toarray()
	assert_true(np.allclose(mat,np.asarray([[ 0.1 ,  0,0       ],[ 0,  0.2,0],[ 0 ,0,0.3       ]], dtype=np.float32)))